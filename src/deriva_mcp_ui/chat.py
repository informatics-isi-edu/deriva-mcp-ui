"""Claude tool-calling loop and SSE streaming.

Public API
----------
run_chat_turn(user_message, session, settings)
    AsyncIterator[dict] -- yields event dicts as Claude produces them.
    Event types:
      {"type": "text",       "content": str}  -- streamed text chunk
      {"type": "tool_start", "name": str, "input": dict}  -- before tool call
      {"type": "tool_end",   "name": str, "result": str}  -- after tool call
    Executes the full tool-calling loop: stream text, dispatch tool_use blocks
    to the MCP server, feed results back, repeat until stop_reason == end_turn.
    Modifies session.history, session.tools, and session.schema_primed in place;
    caller must persist the session.

system_prompt(settings, session, schema_context) -> str
    Returns the operator-configured system prompt, optionally extended with
    schema context injected by _prime_schema on the first turn.

trim_history(messages, max_turns) -> list
    Trims the messages list to at most max_turns user/assistant pairs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import anthropic

from .mcp_client import MCPAuthError, call_tool, get_prompt, list_tools, open_session

if TYPE_CHECKING:
    from .config import Settings
    from .storage.base import Session

logger = logging.getLogger(__name__)

# Maximum tokens to request from Claude per streaming call
_MAX_TOKENS = 8192

# Schema priming: truncate injected context to this many characters.
# Each schema with ~7 tables is roughly 2-5k chars in JSON; 20k allows most catalogs.
_SCHEMA_PRIMING_MAX_CHARS = 20000

# Schemas to skip during priming -- system/internal tables that aren't useful
# for user queries.
_SKIP_SCHEMAS = {"public"}

# Tool result sent to the client for display in the tool call block
_TOOL_RESULT_PREVIEW = 1000

# Tool result fed back to Claude in the current turn.  Keeping this bounded prevents
# large schema/entity responses from blowing the input token budget when combined with
# the (fixed) tool-list cost from the MCP server.  10k chars handles ~10-15 typical
# entity rows with full text columns without truncation.
_TOOL_RESULT_TO_CLAUDE = 10000

# Tool result truncation in stored history -- older turns only need a summary.
_HISTORY_TOOL_RESULT_MAX = 3000

# Retry on transient Anthropic API errors (429 rate-limit, 529 overloaded).
# Only retries are attempted before any text has been yielded in a given loop
# iteration -- once text is in-flight to the client we cannot roll it back.
_MAX_API_RETRIES = 3
_RETRY_BASE_DELAY = 5.0  # seconds; doubles each attempt (5, 10, 20)

# Minimum delay (seconds) between consecutive tool-calling loop iterations when
# the same tool is called again -- catches background-task polling loops where
# Claude says it will wait but cannot actually sleep.
_POLL_DELAY_SECONDS = 5.0


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def system_prompt(
    settings: Settings,
    session: Session,
    schema_context: str = "",
    guide_context: str = "",
    ermrest_syntax: str = "",
) -> str:  # noqa: ARG002
    """Return the system prompt for this session.

    schema_context is injected by _prime_schema on the first turn of a
    default-catalog conversation; empty string means no priming yet.
    guide_context is the concatenated tool guide prompts, injected on the
    first turn so the LLM has behavioral guidance from the start.
    ermrest_syntax is ERMrest URL reference documentation from RAG, injected
    on the first turn so the LLM constructs correct query URLs.
    """
    if settings.default_catalog_mode:
        label = settings.default_catalog_label or settings.default_hostname
        base = (
            f"You are a DERIVA data assistant for the {label} catalog. "
            "You have access to tools for querying and managing this catalog. "
            "When answering questions about data, schema, or annotations, "
            "use the available tools rather than relying on prior knowledge."
        )
    else:
        base = (
            "You are a DERIVA data assistant. "
            "You have access to tools for querying and managing DERIVA catalogs. "
            "When the user wants to work with a specific catalog, ask for the "
            "hostname and catalog ID if they have not been provided."
        )

    # Build mandatory rules -- rule 1 depends on whether schema was injected.
    rules = [
        "\n\nMANDATORY RULES -- violations of these rules waste tool calls and user time:\n"
    ]

    if schema_context:
        rules.append(
            "1. SCHEMA IS ALREADY LOADED. The catalog schema (tables, columns, foreign "
            "keys) is provided below in this system prompt. Do NOT call get_schema, "
            "get_table, get_table_columns, or list_schemas to explore -- that information "
            "is already here. Read it and use it directly."
        )
    else:
        rules.append(
            "1. SCHEMA LOOKUP. Call get_schema ONCE to learn the catalog structure, "
            "then use that information for all subsequent queries. Do NOT call "
            "get_schema, get_table, or get_table_columns repeatedly."
        )

    rules.append(
        "2. USE query_attribute FOR MULTI-TABLE QUERIES. When the user asks for data "
        "that spans related tables, construct a single query_attribute call with a "
        "join path (e.g. Schema:TableA/FK_Column=value/Schema:TableB). Do NOT call "
        "get_entities on individual tables and try to correlate results."
    )
    rules.append(
        "3. ZERO RESULTS IS A COMPLETE ANSWER. If a query returns 0 rows and there "
        "was no HTTP error, report that to the user and stop. Do NOT reformulate "
        "the query, do NOT investigate intermediate tables, do NOT try different "
        "join paths. Zero rows means the data does not exist."
    )
    rules.append(
        "4. DISPLAY ALL COLUMNS WITH FULL VALUES. Always show the RID column. Omit "
        "only the system columns RCT, RMT, RCB, RMB. Show every other column even "
        "if all values are null -- never hide a column. Never truncate, abbreviate, "
        "or shorten any cell value -- display full text verbatim even if it makes "
        "the table wide."
    )
    rules.append(
        "5. When offering options, use a numbered list. When polling background tasks, "
        "wait at least 5 seconds between checks."
    )
    base += "\n".join(rules)

    if schema_context:
        base += (
            "\n\n---\nAvailable schema information (USE THIS -- do not call"
            " get_schema or get_table to re-fetch what is already here):\n"
            + schema_context
        )

    if guide_context:
        base += "\n\n---\nTool usage guides:\n" + guide_context

    if ermrest_syntax:
        base += "\n\n---\nERMrest URL syntax reference:\n" + ermrest_syntax

    return base


# ---------------------------------------------------------------------------
# Guide prompt fetching
# ---------------------------------------------------------------------------

# Names of guide prompts to fetch from the MCP server and inject into the
# system prompt on the first turn.
_GUIDE_PROMPT_NAMES = [
    "query_guide",
    "entity_guide",
    "annotation_guide",
    "catalog_guide",
]


async def _fetch_guides(session: Session, settings: Settings, *, mcp_session=None) -> str:
    """Fetch all guide prompts from the MCP server and return them joined.

    Fetches all guides sequentially on a shared MCP session to avoid
    opening a new connection per prompt. Returns an empty string if none
    succeed.
    """
    mcp_url = settings.mcp_url
    token = session.bearer_token

    parts: list[str] = []
    for name in _GUIDE_PROMPT_NAMES:
        result = await get_prompt(token, name, mcp_url, session=mcp_session)
        if result:
            parts.append(result)
        else:
            logger.debug("Guide prompt %s unavailable", name)

    if parts:
        logger.info("Fetched %d guide prompts (%d chars)", len(parts), sum(len(p) for p in parts))
    else:
        logger.warning("No guide prompts fetched from MCP server")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# ERMrest syntax priming
# ---------------------------------------------------------------------------

# RAG queries to fetch ERMrest URL syntax documentation. Each query targets
# a different aspect of the API so the vector search returns complementary
# chunks rather than duplicates.
_ERMREST_SYNTAX_QUERIES = [
    "ERMrest URL path syntax attribute query filter operators",
    "ERMrest path entity filter predicate join foreign key",
]

# Budget for ERMrest syntax context injected into the system prompt.
_ERMREST_SYNTAX_MAX_CHARS = 6000


async def _prime_ermrest_syntax(session: Session, settings: Settings, *, mcp_session=None) -> str:
    """Fetch ERMrest URL syntax documentation from RAG and return it for injection.

    Runs RAG searches sequentially on a shared MCP session to avoid opening
    a new connection per query. Deduplicates by source and returns the
    concatenated text. Returns an empty string if RAG is unavailable or has
    no relevant docs indexed.
    """
    mcp_url = settings.mcp_url
    token = session.bearer_token

    results: list[str | Exception] = []
    for q in _ERMREST_SYNTAX_QUERIES:
        try:
            r = await call_tool(token, "rag_search", {"query": q, "limit": 5}, mcp_url, session=mcp_session)
            results.append(r)
        except Exception as exc:
            results.append(exc)

    import json as _json
    seen_sources: set[str] = set()
    chunks: list[str] = []
    total_chars = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("ERMrest syntax RAG query %d failed: %s", i, result)
            continue
        if not isinstance(result, str):
            logger.warning("ERMrest syntax RAG query %d returned non-string: %s", i, type(result))
            continue
        if result.startswith("Error:"):
            logger.warning("ERMrest syntax RAG query %d returned error: %s", i, result[:200])
            continue
        try:
            entries = _json.loads(result)
        except Exception:
            continue
        for entry in entries:
            source = entry.get("source", "")
            text = entry.get("text", "")
            if not text or source in seen_sources:
                continue
            seen_sources.add(source)
            if total_chars + len(text) > _ERMREST_SYNTAX_MAX_CHARS:
                break
            chunks.append(text)
            total_chars += len(text)

    if chunks:
        logger.info("ERMrest syntax priming: %d chunks, %d chars", len(chunks), total_chars)
    else:
        logger.warning("ERMrest syntax priming: no docs found in RAG")
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Schema priming
# ---------------------------------------------------------------------------


async def _prime_schema(session: Session, settings: Settings, *, mcp_session=None) -> str:
    """Return schema context for the default catalog to inject into the system prompt.

    Calls get_catalog_info to discover schemas, then get_schema for each to get
    tables and columns. Returns a clean, structured JSON listing that the LLM
    can immediately use for constructing queries.
    """
    import json as _json

    hostname = settings.default_hostname
    catalog_id = settings.default_catalog_id
    mcp_url = settings.mcp_url
    token = session.bearer_token

    # Step 1: get schema names
    try:
        info_text = await call_tool(
            token,
            "get_catalog_info",
            {"hostname": hostname, "catalog_id": catalog_id},
            mcp_url,
            session=mcp_session,
        )
        if not info_text or info_text.startswith("Error:"):
            logger.warning("Schema priming: get_catalog_info failed: %s", info_text[:200] if info_text else "empty")
            return ""
        info = _json.loads(info_text)
        schema_names = [
            s["schema"] for s in info.get("schemas", [])
            if s["schema"] not in _SKIP_SCHEMAS
        ]
    except Exception as exc:
        logger.warning("Schema priming: get_catalog_info failed: %s", exc)
        return ""

    if not schema_names:
        logger.warning("Schema priming: no schemas found in catalog %s/%s", hostname, catalog_id)
        return ""

    # Step 2: get full schema details for each
    parts: list[str] = []
    total_chars = 0
    for schema_name in schema_names:
        try:
            schema_text = await call_tool(
                token,
                "get_schema",
                {"hostname": hostname, "catalog_id": catalog_id, "schema": schema_name},
                mcp_url,
                session=mcp_session,
            )
            if schema_text and not schema_text.startswith("Error:"):
                if parts and total_chars + len(schema_text) > _SCHEMA_PRIMING_MAX_CHARS:
                    logger.debug("Schema priming: budget exceeded at schema %s (%d chars so far)", schema_name, total_chars)
                    break
                parts.append(schema_text)
                total_chars += len(schema_text)
        except Exception as exc:
            logger.debug("Schema priming: get_schema(%s) failed: %s", schema_name, exc)

    if parts:
        result = "\n".join(parts)
        logger.debug("Schema priming: %d schemas, %d chars", len(parts), len(result))
        return result

    logger.warning(
        "Schema priming failed for %s/%s -- proceeding without context", hostname, catalog_id
    )
    return ""


# ---------------------------------------------------------------------------
# History trimming
# ---------------------------------------------------------------------------


def trim_history(messages: list[dict[str, Any]], max_turns: int) -> list[dict[str, Any]]:
    """Return messages trimmed to at most max_turns user/assistant pairs.

    Each pair consumes two list entries; we keep the most recent ones.
    Avoids splitting mid-tool-call by advancing the tail forward until it
    starts on a plain user text message -- never on an assistant message or
    a user message that contains tool_result blocks (which would be orphaned
    without the preceding tool_use blocks that were trimmed away).
    """
    limit = max_turns * 2
    if len(messages) <= limit:
        return messages
    tail = messages[-limit:]
    while tail:
        first = tail[0]
        if first.get("role") != "user":
            # landed on an assistant message -- skip it
            tail = tail[1:]
        elif isinstance(first.get("content"), list) and any(
            b.get("type") == "tool_result" for b in first["content"]
        ):
            # landed on a tool_result continuation whose tool_use was trimmed --
            # drop it and the following assistant reply (keeps pairs balanced)
            tail = tail[2:]
        else:
            break
    return tail


# ---------------------------------------------------------------------------
# Tool-calling loop
# ---------------------------------------------------------------------------


async def run_chat_turn(
    user_message: str,
    session: Session,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    """Run one chat turn and yield event dicts as Claude produces output.

    Yields:
      {"type": "status",     "message": str}               -- priming status update
      {"type": "text",       "content": str}               -- streamed text chunk
      {"type": "tool_start", "name": str, "input": dict}   -- before each tool call
      {"type": "tool_end",   "name": str, "result": str}   -- after each tool call

    Mutates session.history (appends the completed turn), session.tools
    (caches on first call), and session.schema_primed (set after first-turn
    priming).  The caller must persist the session after the iterator is
    exhausted.

    Raises MCPAuthError if the MCP server rejects the bearer token.
    """
    mcp_url = settings.mcp_url

    # First turn: open a single MCP session and batch all priming calls
    # (list_tools, guide prompts, schema, ERMrest syntax) on it to avoid
    # the per-call connection overhead (each connection does a full MCP
    # initialize handshake including ListToolsRequest).
    needs_tools = session.tools is None
    needs_priming = not session.schema_primed

    if needs_tools or needs_priming:
        yield {"type": "status", "message": "Connecting to server..."}
        async with open_session(session.bearer_token, mcp_url) as mcp_sess:
            if needs_tools:
                session.tools = await list_tools(session.bearer_token, mcp_url, session=mcp_sess)
                logger.debug("Cached %d tools for session %s", len(session.tools), session.user_id)

            if needs_priming:
                if settings.default_catalog_mode:
                    yield {"type": "status", "message": "Loading catalog schema..."}
                    session.primed_schema = await _prime_schema(session, settings, mcp_session=mcp_sess)
                yield {"type": "status", "message": "Loading tool guides..."}
                session.primed_guides = await _fetch_guides(session, settings, mcp_session=mcp_sess)
                session.primed_ermrest = await _prime_ermrest_syntax(session, settings, mcp_session=mcp_sess)
                session.schema_primed = True

    # Use cached priming context on every turn (persisted in session)
    schema_context = session.primed_schema
    guide_context = session.primed_guides
    ermrest_syntax = session.primed_ermrest

    if needs_priming:
        logger.info(
            "Priming complete: schema=%d chars, guides=%d chars, ermrest=%d chars",
            len(schema_context), len(guide_context), len(ermrest_syntax),
        )
        if not schema_context:
            logger.warning("Schema context is EMPTY -- LLM will not have schema in system prompt")

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    prompt = system_prompt(settings, session, schema_context, guide_context, ermrest_syntax)

    # Prompt caching: mark the system prompt and the tail of the tool list as
    # cacheable so that repeated calls within the 5-minute cache TTL reuse the
    # tokenized blocks rather than re-processing them on every turn.
    system_block: list[dict[str, Any]] = [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}
    ]
    tools_with_cache: list[dict[str, Any]] = list(session.tools)
    if tools_with_cache:
        tools_with_cache[-1] = {**tools_with_cache[-1], "cache_control": {"type": "ephemeral"}}

    messages: list[dict[str, Any]] = list(session.history) + [
        {"role": "user", "content": user_message}
    ]

    prev_tool_names: set[str] = set()  # tools called in the previous loop iteration

    while True:
        stream_kwargs: dict[str, Any] = dict(
            model=settings.claude_model,
            max_tokens=_MAX_TOKENS,
            system=system_block,
            tools=tools_with_cache,
            messages=messages,
        )

        response = None
        for attempt in range(_MAX_API_RETRIES + 1):
            text_yielded = False
            try:
                async with client.messages.stream(**stream_kwargs) as stream:
                    async for text in stream.text_stream:
                        text_yielded = True
                        yield {"type": "text", "content": text}
                    response = await stream.get_final_message()
                break  # success -- exit retry loop
            except anthropic.RateLimitError:
                if text_yielded or attempt >= _MAX_API_RETRIES:
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.0fs",
                    attempt + 1, _MAX_API_RETRIES, delay,
                )
                await asyncio.sleep(delay)
            except anthropic.APIStatusError as exc:
                if exc.status_code != 529 or text_yielded or attempt >= _MAX_API_RETRIES:
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "API overloaded (attempt %d/%d), retrying in %.0fs",
                    attempt + 1, _MAX_API_RETRIES, delay,
                )
                await asyncio.sleep(delay)

        if response.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": _content_to_dicts(response.content)})
            break

        if response.stop_reason != "tool_use":
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            messages.append({"role": "assistant", "content": _content_to_dicts(response.content)})
            break

        # Execute tool_use blocks
        curr_tool_names = {b.name for b in response.content if b.type == "tool_use"}
        if curr_tool_names & prev_tool_names:
            # Same tool(s) called again -- enforce a real delay so background-task
            # polling loops actually wait rather than re-polling immediately.
            logger.debug(
                "Poll delay %.0fs: repeated tools %s",
                _POLL_DELAY_SECONDS,
                curr_tool_names & prev_tool_names,
            )
            await asyncio.sleep(_POLL_DELAY_SECONDS)
        prev_tool_names = curr_tool_names

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            yield {"type": "tool_start", "name": block.name, "input": block.input}
            try:
                result_text = await call_tool(
                    session.bearer_token, block.name, block.input, mcp_url
                )
            except MCPAuthError:
                raise
            except Exception as exc:
                logger.error("Tool %s failed: %s", block.name, exc)
                result_text = f"Error executing tool {block.name}: {exc}"
            yield {"type": "tool_end", "name": block.name, "result": result_text[:_TOOL_RESULT_PREVIEW]}

            claude_content = result_text[:_TOOL_RESULT_TO_CLAUDE]
            if len(result_text) > _TOOL_RESULT_TO_CLAUDE:
                claude_content += "\n[result truncated]"
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": claude_content}
            )

        messages.append({"role": "assistant", "content": _content_to_dicts(response.content)})
        messages.append({"role": "user", "content": tool_results})

    full_history = _truncate_history_tool_results(messages)
    session.full_history = full_history
    session.history = trim_history(full_history, settings.max_history_turns)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_history_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return messages with tool_result content truncated to _HISTORY_TOOL_RESULT_MAX chars.

    The full result is used within the current turn (fed back to Claude), but replaying
    large tool outputs in every subsequent request wastes input tokens.  This truncation
    only affects the stored history copy, not the live messages list.
    """
    out = []
    for msg in messages:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            out.append(msg)
            continue
        new_content = []
        changed = False
        for block in msg["content"]:
            if (
                block.get("type") == "tool_result"
                and len(block.get("content", "")) > _HISTORY_TOOL_RESULT_MAX
            ):
                new_content.append(
                    {**block, "content": block["content"][:_HISTORY_TOOL_RESULT_MAX] + "\n[truncated]"}
                )
                changed = True
            else:
                new_content.append(block)
        out.append({**msg, "content": new_content} if changed else msg)
    return out


_BLOCK_FIELDS: dict[str, list[str]] = {
    "text": ["type", "text"],
    "tool_use": ["type", "id", "name", "input"],
    "thinking": ["type", "thinking", "signature"],
    "redacted_thinking": ["type", "data"],
}


def _content_to_dicts(content: list[Any]) -> list[dict[str, Any]]:
    """Convert Anthropic SDK content block objects to plain dicts.

    Only the fields the Messages API accepts per block type are kept;
    SDK-internal fields (e.g. parsed_output) are stripped so that
    replaying history does not trigger API validation errors.
    """
    result = []
    for block in content:
        data: dict[str, Any] = block.model_dump() if hasattr(block, "model_dump") else dict(block)
        block_type = data.get("type", "")
        allowed = _BLOCK_FIELDS.get(block_type)
        if allowed:
            result.append({k: data[k] for k in allowed if k in data})
        else:
            result.append(data)
    return result
