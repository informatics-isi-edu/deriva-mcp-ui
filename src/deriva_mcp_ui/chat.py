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

from .mcp_client import MCPAuthError, call_tool, list_tools

if TYPE_CHECKING:
    from .config import Settings
    from .storage.base import Session

logger = logging.getLogger(__name__)

# Maximum tokens to request from Claude per streaming call
_MAX_TOKENS = 8192

# Schema priming: truncate injected context to this many characters (~1k tokens)
_SCHEMA_PRIMING_MAX_CHARS = 4000

# Tool result sent to the client for display in the tool call block
_TOOL_RESULT_PREVIEW = 1000

# Tool result fed back to Claude in the current turn.  Keeping this bounded prevents
# large schema/entity responses from blowing the input token budget when combined with
# the (fixed) tool-list cost from the MCP server.
_TOOL_RESULT_TO_CLAUDE = 6000

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


def system_prompt(settings: Settings, session: Session, schema_context: str = "") -> str:  # noqa: ARG002
    """Return the system prompt for this session.

    schema_context is injected by _prime_schema on the first turn of a
    default-catalog conversation; empty string means no priming yet.
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

    base += (
        " When offering the user a list of options or next steps, always use a "
        "numbered list so the user can reply with a number to select an option."
        " When polling for background task status, wait at least 5 seconds between"
        " each status check."
        " When displaying query results: always include the RID column; omit the"
        " ERMrest system columns RCT, RMT, RCB, RMB unless the user explicitly asks"
        " for them; show ALL other columns including those whose values are entirely"
        " null -- never hide or drop a column just because its values are null."
    )

    if schema_context:
        return base + "\n\n---\nAvailable schema information:\n" + schema_context

    return base


# ---------------------------------------------------------------------------
# Schema priming
# ---------------------------------------------------------------------------


async def _prime_schema(session: Session, settings: Settings) -> str:
    """Return schema context for the default catalog to inject into the system prompt.

    Tries rag_search first (semantic, concise).  Falls back to get_schema if
    rag_search fails or returns an error result.  Truncates to
    _SCHEMA_PRIMING_MAX_CHARS to stay within a reasonable token budget.
    """
    hostname = settings.default_hostname
    catalog_id = settings.default_catalog_id
    mcp_url = settings.mcp_url
    token = session.bearer_token

    # Attempt 1: rag_search
    try:
        text = await call_tool(
            token,
            "rag_search",
            {"query": "tables columns schema", "hostname": hostname, "catalog_id": catalog_id},
            mcp_url,
        )
        if text and not text.startswith("Error:"):
            logger.debug("Schema priming via rag_search: %d chars", len(text))
            return text[:_SCHEMA_PRIMING_MAX_CHARS]
    except Exception as exc:
        logger.debug("rag_search unavailable for priming: %s", exc)

    # Attempt 2: get_schema
    try:
        text = await call_tool(
            token,
            "get_schema",
            {"hostname": hostname, "catalog_id": catalog_id},
            mcp_url,
        )
        if text and not text.startswith("Error:"):
            logger.debug("Schema priming via get_schema: %d chars", len(text))
            return text[:_SCHEMA_PRIMING_MAX_CHARS]
    except Exception as exc:
        logger.debug("get_schema unavailable for priming: %s", exc)

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
      {"type": "text",       "content": str}          -- streamed text chunk
      {"type": "tool_start", "name": str, "input": dict}  -- before each tool call
      {"type": "tool_end",   "name": str, "result": str}  -- after each tool call

    Mutates session.history (appends the completed turn), session.tools
    (caches on first call), and session.schema_primed (set after first-turn
    priming).  The caller must persist the session after the iterator is
    exhausted.

    Raises MCPAuthError if the MCP server rejects the bearer token.
    """
    mcp_url = settings.mcp_url

    # Cache tool list on first turn
    if session.tools is None:
        session.tools = await list_tools(session.bearer_token, mcp_url)
        logger.debug("Cached %d tools for session %s", len(session.tools), session.user_id)

    # Schema priming: inject catalog schema on the first turn of a default-catalog session
    schema_context = ""
    if settings.default_catalog_mode and not session.schema_primed:
        schema_context = await _prime_schema(session, settings)
        session.schema_primed = True

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    prompt = system_prompt(settings, session, schema_context)

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

    session.history = trim_history(
        _truncate_history_tool_results(messages), settings.max_history_turns
    )


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
