"""LLM tool-calling loop, RAG-only mode, and SSE streaming.

Public API
----------
run_chat_turn(user_message, session, settings)
    AsyncIterator[dict] -- yields event dicts as the LLM produces them.
    Event types:
      {"type": "text",       "content": str}  -- streamed text chunk
      {"type": "tool_start", "name": str, "input": dict}  -- before tool call
      {"type": "tool_end",   "name": str, "result": str}  -- after tool call
    Routes to either the full LLM tool-calling loop (via LiteLLM) or the
    RAG-only response path depending on settings.operating_tier.
    Modifies session.history, session.tools, and session.schema_primed in
    place; caller must persist the session.

system_prompt(settings, session, schema_context) -> str
    Returns the operator-configured system prompt, optionally extended with
    schema context injected by _prime_schema on the first turn.

trim_history(messages, max_turns) -> list
    Trims the messages list to at most max_turns user exchanges.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import litellm

from .audit import audit_event
from .mcp_client import MCPAuthError, call_tool, get_prompt, list_tools, open_session

if TYPE_CHECKING:
    from .config import Settings
    from .storage.base import Session

logger = logging.getLogger(__name__)

# Maximum tokens to request from the LLM per streaming call
_MAX_TOKENS = 8192

# Schema priming: truncate injected context to this many characters.
# Each schema with ~7 tables is roughly 2-5k chars in JSON; 20k allows most catalogs.
_SCHEMA_PRIMING_MAX_CHARS = 20000

# Schemas to skip during priming -- system/internal tables that aren't useful
# for user queries.
_SKIP_SCHEMAS = {"public"}

# Tool result sent to the client for display in the tool call block
_TOOL_RESULT_PREVIEW = 1000

# Tool result fed back to the LLM in the current turn.  Keeping this bounded
# prevents large schema/entity responses from blowing the input token budget
# when combined with the (fixed) tool-list cost from the MCP server.  10k chars
# handles ~10-15 typical entity rows with full text columns without truncation.
_TOOL_RESULT_TO_LLM = 10000

# Tool result truncation in stored history -- older turns only need a summary.
_HISTORY_TOOL_RESULT_MAX = 3000

# Retry on transient LLM API errors (429 rate-limit, 529 overloaded).
# Only retries are attempted before any text has been yielded in a given loop
# iteration -- once text is in-flight to the client we cannot roll it back.
_MAX_API_RETRIES = 3
_RETRY_BASE_DELAY = 5.0  # seconds; doubles each attempt (5, 10, 20)

# Minimum delay (seconds) between consecutive tool-calling loop iterations when
# the same tool is called again -- catches background-task polling loops where
# the LLM says it will wait but cannot actually sleep.
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
        hostname = settings.default_hostname
        catalog_id = settings.default_catalog_id
        base = (
            f"You are a DERIVA data assistant for the {label} catalog. "
            "You have access to tools for querying and managing this catalog. "
            "When answering questions about data, schema, or annotations, "
            "use the available tools rather than relying on prior knowledge. "
            f"For EVERY tool call that accepts hostname and catalog_id parameters, "
            f"you MUST pass hostname=\"{hostname}\" and catalog_id=\"{catalog_id}\" "
            f"exactly as written. Never omit these arguments or substitute a different value."
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

        "4B. **INLINE IMAGE DISPLAY**:"
        "When you retrieve URLs pointing to image files (jpg, png, gif, etc.) from" 
        "Hatrac or other DERIVA file storage:"
        "1. Format them as Markdown image links: ![alt text](url)"
        "2. Use the filename or a descriptive caption as the alt text"
        "3. Display images inline in your response for visual reference"
        "4. This applies to thumbnails, previews, and any publicly accessible images"
        "5. You should look for tables with names like 'thumbnails' or 'previews' and prioritize the display of those"
        " before other file tables."
        "6. IMPORTANT: if the returned URL does not contain a scheme/host/port prepend the scheme/host/port of the catalog"
        "the results were returned from."
    )
    rules.append(
        "5. ALWAYS USE NUMBERED LISTS FOR OPTIONS AND FOLLOW-UPS. Any time you "
        "present options, suggested next steps, or follow-up actions -- including "
        "after a tool result -- format them as a numbered list (1. ... 2. ... 3. ...). "
        "If the user replies with a bare number after you presented a numbered list, "
        "interpret it as selecting that option -- not as a literal quantity. "
        "When polling background tasks, wait at least 5 seconds between checks."
    )
    rules.append(
        "RULE 6: COMMUNICATION STYLE"
        "Be direct and factual. Present findings without meta-commentary about your actions."
        "DO:"
        "- State results directly: \"3 datasets match the criteria\" (not \"I found 3 datasets\")"
        "- Use numbers as headers in lists: \"1. Dataset A  2. Dataset B\""
        "- Brief connectors: \"The results show...\", \"This means...\", \"Therefore...\""
        "- Necessary context: \"0 rows — the data does not exist\""
        "- Spare enthusiasm only for genuinely surprising findings: rare unexpected correlations, large recovered datasets thought lost"
        "DON'T:"
        "- Narrate steps: \"Let me retrieve...\", \"Now I'll query...\", \"First, I searched...\""
        "- Use exclamation marks except in rare surprising findings"
        "- Say \"Great!\", \"Perfect!\", \"Excellent!\", \"Wonderful!\" — just present the data"
        "- Soften bad news: \"Unfortunately...\" — state it plainly"
        "- Use filler: \"As you can see...\", \"Interestingly...\""
        "Transitions between multi-turn responses:"
        "- Omit: \"Now let me address your follow-up question\""
        "- Use: \"Regarding your second question:\" or just answer directly"
        "Length and truncation:"
        "- Never abbreviate or hide columns/values for brevity"
        "- For very long fields (>500 chars), render in full but note if truncated in display only"
    )
    rules.append(
        "7. RAG SEARCH RESULT FORMAT. When presenting rag_search results, use a NUMBERED "
        "list (1. 2. 3. ...) ordered by relevance score descending. For EVERY result item: "
        "(a) if the result JSON contains a non-empty \"url\" field, make the result heading "
        "a Markdown hyperlink using that URL -- e.g. "
        "\"1. **[Dataset RID=2B8P](https://...)** (relevance: 0.75)\"; "
        "(b) include a one-sentence italic summary (in *italics*) directly under the heading "
        "describing what that source says about the question. "
        "Never omit the number, link, or summary."
    )
    rules.append(
        "**RULE 7B: CATALOG QUERY RESULT FORMAT.** When presenting results from `query_attribute`, `get_entities`, "
        " `count_table`, or similar direct catalog queries, **always hyperlink entity records back to the catalog UI**:"
        "- For dataset records: use the Chaise detail view URL format (EXAMPLE): "
        "   `https://staging.facebase.org/chaise/record/#1/isa:dataset/RID={RID}`"
        "- For other table records: use (EXAMPLE) `https://staging.facebase.org/chaise/record/#1/{schema}:{table}/RID={RID}`"
        "- Format the table name (accession, title, or primary identifier) as the link text in Markdown: `[{display_text}]({url})`"
        "- Include a numbered list (1. 2. 3. ...) when presenting multiple records"
        "- Always include the RID in the display (in addition to other relevant columns) so users can verify and cite the record"
        "Example format:"
        "1. **[Dataset Title](https://staging.facebase.org/chaise/record/#1/isa:dataset/RID=1-T8KE)**"
        "   (RID: 1-T8KE, Accession: FB00001113, Released: Aug 13, 2020)"
        "   *Optional: one-sentence summary of the record*"
        ""
        "This rule should apply whenever the catalog context (hostname + catalog_id) is known."
    )
    rules.append(
        "8. INLINE IMAGE DISPLAY. When you retrieve image file URLs from the catalog"
        "(from thumbnail, file, or preview tables), render them inline using"
        "Markdown image syntax: ![alt-text](url). Do not just provide links—display"
        "the images so the user can see them directly in the response."
        "You should look for tables with names like 'thumbnails' or 'previews' and prioritize the display of those"
        " before other file tables. Prioritize lookups on thumbnail tables first."
        "IMPORTANT: if the returned URL does not contain a scheme/host/port prepend the scheme/host/port of the catalog"
        "the results were returned from."
    )
    rules.append(
        "TOOL SELECTION PRIORITY:"
        "When the user asks a question, follow this priority order:"
        "a) DEFINITION / EXPLANATION QUESTIONS → Use your knowledge"
        "Examples: 'what is craniosynostosis', 'explain ERMrest', 'how does premature suture fusion work', "
        "'what is a foreign key' -- Feel free to augment your knowledge with rag_search, if the question looks like it"
        "might generate a hit in the known rag sources."
        "b) CATALOG-SPECIFIC DATA QUESTIONS → STOP. IMPORTANT: rag_search first !"
        "Examples: 'what datasets exist for x', 'show me y datasets', 'what research projects study z'"
        "Use rag_search to discover what data exists, THEN follow up with query_attribute or get_entities if you need "
        "specific records."
        "b) PROGRAMMING-SPECIFIC DATA QUESTIONS → rag_search first"
        "When a user asks how to do any kind of API or programmatic access, or when you are suggesting it, "
        "use rag_search first, since this may be more recent than your domain knowledge. Then feel free to augment with "
        "your own knowledge."        
        "c) SPECIFIC DATA RETRIEVAL → query_attribute or get_entities"
        "When you already know: a specific RID, an exact filter value, or which table contains the data. Go directly "
        "to the query without exploring first."
        "d) MULTI-TABLE JOINS → query_attribute with join path"
        "When the user asks for data across related tables. Use the schema context already provided to construct a "
        "single join path—do NOT call get_entities on individual tables separately."
        "KNOWLEDGE VS. CATALOG DATA:"
        "YOUR KNOWLEDGE: Use freely for general definitions, concepts, methods, syntax explanations, domain expertise,"
        " and background context. You do not need rag_search to answer 'what is X' or 'how does Y work.'"
        "CATALOG DATA: Search or query only when the user asks about specific FaceBase content, datasets, records, "
        "or what data exists in the catalog."
        "DO NOT use rag_search for: general definitions or explanations, basic DERIVA/ERMrest syntax questions, "
        "schema exploration (it's already provided above), 'How do I do X in general' questions, or standard "
        "terminology from any field."
        "The schema is already loaded in the system prompt. You have all table names, columns, and foreign key "
        "relationships. Proceed directly to writing queries—do not make schema-fetching calls (get_schema, get_table, "
        "list_schemas, etc.)."
    )
    base += "\n".join(rules)

    if schema_context:
        base += (
            "\n\nAvailable schema information (USE THIS -- do not call"
            " get_schema or get_table to re-fetch what is already here):\n"
            + schema_context
        )

    if guide_context:
        base += "\n\nTool usage guides:\n" + guide_context

    if ermrest_syntax:
        base += "\n\nERMrest URL syntax reference:\n" + ermrest_syntax

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
    mcp_url = settings.remap_url(settings.mcp_url)
    token = session.bearer_token

    parts: list[str] = []
    for name in _GUIDE_PROMPT_NAMES:
        result = await get_prompt(token, name, mcp_url, session=mcp_session, ssl_verify=settings.ssl_verify)
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
    mcp_url = settings.remap_url(settings.mcp_url)
    token = session.bearer_token

    results: list[str | Exception] = []
    for q in _ERMREST_SYNTAX_QUERIES:
        try:
            r = await call_tool(token, "rag_search", {"query": q, "limit": 5}, mcp_url, session=mcp_session, ssl_verify=settings.ssl_verify)
            results.append(r)
        except Exception as exc:
            results.append(exc)

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
            entries = json.loads(result)
        except Exception:
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
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
    hostname = settings.default_hostname
    catalog_id = settings.default_catalog_id
    mcp_url = settings.remap_url(settings.mcp_url)
    token = session.bearer_token

    # Step 1: get schema names
    try:
        info_text = await call_tool(
            token,
            "get_catalog_info",
            {"hostname": hostname, "catalog_id": catalog_id},
            mcp_url,
            session=mcp_session,
            ssl_verify=settings.ssl_verify,
        )
        if not info_text or info_text.startswith("Error:"):
            logger.warning("Schema priming: get_catalog_info failed: %s", info_text[:200] if info_text else "empty")
            return ""
        info = json.loads(info_text)
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
                ssl_verify=settings.ssl_verify,
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
# RAG-only response path
# ---------------------------------------------------------------------------

# Maximum number of RAG results to request per query.
# Fetch more chunks than we display so per-type capping has headroom to find
# lower-ranking results from under-represented source types.  No LLM context
# window to worry about in RAG-only mode, so retrieve more for better
# per-source field coverage (project/phenotype alongside the title chunk).
_RAG_SEARCH_LIMIT = 100

# Per-source-type cap: prevents one high-scoring type (e.g. enriched dataset records)
# from consuming all result slots and hiding lower-scoring results of other types.
# Applied in score order, so the best N of each type are always shown.
_PER_TYPE_CAP: dict[str, int] = {
    "catalog-data": 5,  # catalog dataset records
    "schema": 2,        # schema index results
}
_PER_TYPE_CAP_DEFAULT = 5  # web crawl, documentation, GitHub sources, etc.


# Regex patterns for question-type detection
import re

_RE_WHAT_IS = re.compile(
    r"\b(what\s+(is|are|does|do)\b|define\b|describe\b|explain\b)", re.IGNORECASE,
)
_RE_HOW_TO = re.compile(
    r"\b(how\s+(do|can|to|should)\b|steps?\s+to\b|process\b|method\b)", re.IGNORECASE,
)
_RE_HOW_MANY = re.compile(
    r"\b(how\s+(many|much)\b|count\b|number\s+of\b)", re.IGNORECASE,
)
_RE_WHERE = re.compile(r"\bwhere\b", re.IGNORECASE)
_RE_DATA = re.compile(
    r"\b(data|dataset|download|access|file|image|export|import)\b",
    re.IGNORECASE,
)
_RE_SCHEMA = re.compile(
    r"\b(tables?|columns?|schemas?|catalogs?|foreign\s*keys?|fk|primary\s*keys?|pk)\b",
    re.IGNORECASE,
)
_RE_SHOW_ALL = re.compile(
    r"\b(?:show\s+all(?:\s+results?)?|show\s+more(?:\s+results?)?|all\s+results?|more\s+results?|less\s+relevant|lower\s+relevance)\b",
    re.IGNORECASE,
)

# Minimum sentence length to consider meaningful
_MIN_SENTENCE_LEN = 20
# Minimum relevance score to include a RAG result (0-1 scale, cosine similarity)
_MIN_RELEVANCE_SCORE = 0.30
# Default threshold: only show sources at or above this score; lower-scored
# sources are noted but hidden unless the user asks to see them.
_HIGH_RELEVANCE_THRESHOLD = 0.50


# Stop words to strip when extracting key terms
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "after", "before", "above", "below", "and", "or", "but",
    "not", "no", "if", "then", "than", "so", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "you", "your", "he", "she",
    "they", "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "some", "any",
    "work", "works", "tell", "show", "list", "get", "give", "make",
})


def _extract_key_terms(question: str) -> str:
    """Extract technical key terms from a question for a focused secondary search.

    Strips common stop words and question framing to isolate the specific
    technical concepts the user is asking about.  Returns an empty string if
    no meaningful terms remain (in which case a secondary search is skipped).
    """
    # Remove punctuation, lowercase, split into words
    words = re.sub(r"[^\w\s]", " ", question.lower()).split()
    terms = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    if len(terms) <= 1:
        return ""
    # If the terms are very similar to the original question, skip
    if len(terms) >= len(words) - 2:
        return ""
    return " ".join(terms)


def _merge_rag_results(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge primary and secondary RAG search results, deduplicating by record identity.

    For chunks that carry a URL (e.g. enriched catalog records where every row
    has a unique Chaise URL), the URL is the dedup key so all records from the
    same source table are preserved.  For chunks without a URL (docs, schema),
    the source key is used.

    The merged list is sorted by score descending.
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for r in primary:
        key = r.get("url") or r.get("source", "")
        if key not in seen:
            seen.add(key)
            merged.append(r)
    for r in secondary:
        key = r.get("url") or r.get("source", "")
        if key not in seen:
            seen.add(key)
            merged.append(r)
    merged.sort(key=lambda r: float(r.get("score", 0)), reverse=True)
    return merged


async def _rag_only_response(
    user_message: str,
    session: Session,
    settings: Settings,
) -> AsyncIterator[dict[str, Any]]:
    """Serve a response from the RAG subsystem without an LLM.

    Calls rag_search (and optionally get_schema/get_catalog_info for schema
    questions) via the MCP server, then formats the results with source
    citations.

    Yields the same event dict format as run_chat_turn so the SSE layer and
    UI are unchanged.
    """
    mcp_url = settings.remap_url(settings.mcp_url)
    token = session.bearer_token

    # If the message is purely a show-all command (nothing meaningful after
    # stripping the show-all phrase), re-run the previous user query with the
    # threshold lowered instead of searching for "show all results".
    show_all = bool(_RE_SHOW_ALL.search(user_message))
    effective_query = user_message
    if show_all:
        remainder = _RE_SHOW_ALL.sub("", user_message).strip().strip('"\'.,:;!?')
        if len(remainder) < 5:
            for msg in reversed(list(session.history)):
                if msg.get("role") == "user" and not _RE_SHOW_ALL.search(
                    msg.get("content", "")
                ):
                    effective_query = msg["content"]
                    break

    # Schema keyword routing: if the question is clearly about schema, also
    # call get_schema / get_catalog_info for a structured answer.
    schema_text = ""
    if _RE_SCHEMA.search(effective_query) and settings.default_catalog_mode:
        yield {"type": "status", "message": "Looking up schema..."}
        try:
            schema_text = await call_tool(
                token,
                "get_catalog_info",
                {
                    "hostname": settings.default_hostname,
                    "catalog_id": settings.default_catalog_id,
                },
                mcp_url,
                ssl_verify=settings.ssl_verify,
            )
        except Exception as exc:
            logger.debug("RAG-only schema lookup failed: %s", exc)

    # RAG search -- run the full question as the primary query, then
    # extract technical key terms and run a focused secondary query to
    # catch specific API/concept docs that the broad search may miss.
    # Use a higher limit in RAG-only mode: no LLM context window cost, and
    # more chunks means better per-source field coverage.
    yield {"type": "status", "message": "Searching documentation..."}
    rag_results: list[dict[str, Any]] = []
    try:
        result_text = await call_tool(
            token,
            "rag_search",
            {"query": effective_query, "limit": _RAG_SEARCH_LIMIT},
            mcp_url,
            ssl_verify=settings.ssl_verify,
        )
        if result_text and not result_text.startswith("Error:"):
            rag_results = json.loads(result_text)
    except MCPAuthError:
        raise
    except Exception as exc:
        logger.warning("RAG-only search failed: %s", exc, exc_info=True)

    # Multi-query: extract key terms and run a secondary search to boost
    # specific technical documentation that the broad query may rank low.
    key_terms = _extract_key_terms(effective_query)
    if key_terms:
        try:
            term_text = await call_tool(
                token,
                "rag_search",
                {"query": key_terms, "limit": _RAG_SEARCH_LIMIT},
                mcp_url,
                ssl_verify=settings.ssl_verify,
            )
            if term_text and not term_text.startswith("Error:"):
                term_results = json.loads(term_text)
                rag_results = _merge_rag_results(rag_results, term_results)
        except MCPAuthError:
            raise
        except Exception:
            pass  # secondary search is best-effort

    # Supplemental web/doc search: the primary query may return only enriched
    # catalog records if they all score above web sources.  Run a targeted search
    # limited to web-content and user-guide doc types so those sources always
    # have a chance to surface.
    for _web_doc_type in ("web-content", "user-guide"):
        try:
            _web_text = await call_tool(
                token,
                "rag_search",
                {"query": effective_query, "limit": 5, "doc_type": _web_doc_type},
                mcp_url,
                ssl_verify=settings.ssl_verify,
            )
            if _web_text and not _web_text.startswith("Error:"):
                _web_results = json.loads(_web_text)
                rag_results = _merge_rag_results(rag_results, _web_results)
        except MCPAuthError:
            raise
        except Exception:
            pass  # best-effort

    response = _format_rag_response(effective_query, rag_results, schema_text, show_all=show_all)
    yield {"type": "text", "content": response}

    # Build per-turn summary for the audit layer.
    rag_docs = [r.get("url") or r.get("source", "") for r in rag_results]
    rag_scores = [float(r.get("score", 0)) for r in rag_results]
    summary: dict[str, Any] = {
        "tools_invoked": ["rag_search"],
        "rag_triggered": True,
        "rag_document_count": len(rag_results),
        "rag_documents": rag_docs,
        "rag_scores": rag_scores,
        "model": None,
    }
    if settings.audit_diagnostic:
        summary["user_query"] = user_message
        _resp_max = settings.audit_diagnostic_response_max_chars
        summary["response_text"] = response[:_resp_max] if _resp_max > 0 else response
        summary["response_compressed"] = base64.b64encode(gzip.compress(response.encode())).decode()
        summary["tool_inputs"] = [{"name": "rag_search", "args": {"query": effective_query}}]
        summary["tool_outputs"] = []  # formatted response already captures the substance
    yield {"type": "turn_summary", **summary}

    # Store in history
    messages = list(session.history) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response},
    ]
    session.history = trim_history(messages, settings.max_history_turns)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking on decimal numbers or common
    abbreviations (e.g. E13.5, Fig. 3, vs.)."""
    return re.split(r"(?<!\d)[.!?]+(?!\d)(?!\s*[a-z])", text)


def _format_rag_response(
    question: str,
    results: list[dict[str, Any]],
    schema_text: str = "",
    show_all: bool = False,
) -> str:
    """Format RAG search results into a Markdown response organized by source.

    Results are grouped by source, ordered from most to least relevant.
    Each source gets an italic heading with its name and relevance score.
    Structured content (markdown tables, code blocks) is included verbatim;
    prose is sentence-extracted and filtered by question type.

    When show_all is False (default), only sources at or above
    _HIGH_RELEVANCE_THRESHOLD are shown; a note at the end lists how many
    lower-relevance sources are available.
    """
    parts: list[str] = []

    # Include schema info if available
    if schema_text:
        parts.append(schema_text)
        if results:
            parts.append("")  # blank line separator

    # Filter out low-relevance results -- they add noise without value
    results = [
        r for r in results
        if isinstance(r, dict) and float(r.get("score", 0)) >= _MIN_RELEVANCE_SCORE
    ]

    if not results:
        if schema_text:
            return "\n".join(parts)
        return (
            "No relevant documentation found for your question. "
            "Try rephrasing, or ask about a specific table or schema."
        )

    # Low-confidence warning
    top_score = max((float(r.get("score", 0)) for r in results), default=0)
    if top_score < 0.40:
        parts.append(
            "I found some information, but I'm not entirely certain it "
            "fully answers your question.\n"
        )

    # Question-type framing -- use h4 so it groups visually with the source heading
    if _RE_HOW_TO.search(question):
        parts.append("#### Documentation for this process:")
    elif _RE_DATA.search(question):
        parts.append("#### Regarding data access:")
    else:
        parts.append("#### Based on available documentation:")

    # Group results by group key.
    # Enriched dataset records share a compound source key but each has a unique
    # URL (Chaise record page).  Grouping by URL ensures each dataset record
    # gets its own section.  For chunks without a URL, fall back to source key.
    # group_sources maps group_key -> canonical source key (for label derivation)
    # group_doc_types maps group_key -> doc_type tag (used for type cap + threshold exemption)
    group_sources: dict[str, str] = {}
    group_doc_types: dict[str, str] = {}
    source_groups: dict[str, list[dict[str, Any]]] = {}
    source_scores: dict[str, float] = {}
    source_urls: dict[str, str] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source", "unknown")
        url = entry.get("url", "")
        group_key = url if url else source
        score = float(entry.get("score", 0.0))
        group_sources.setdefault(group_key, source)
        _dt = entry.get("doc_type", "") or ("catalog-data" if source.startswith("enriched:") else "")
        group_doc_types.setdefault(group_key, _dt)
        source_groups.setdefault(group_key, []).append(entry)
        if score > source_scores.get(group_key, 0.0):
            source_scores[group_key] = score
            if url:
                source_urls[group_key] = url

    # Order sources by best score (most relevant first)
    ordered_sources = sorted(
        source_scores.keys(), key=lambda s: source_scores[s], reverse=True,
    )

    if show_all:
        # User explicitly asked for everything -- bypass both cap and threshold.
        shown_sources = ordered_sources
        hidden_sources = []
    else:
        # Apply per-type cap (in score order) so one high-scoring type cannot consume
        # all result slots.  Sources beyond the cap are demoted to hidden regardless
        # of their score.
        type_counts: dict[str, int] = {}
        within_cap: list[str] = []
        beyond_cap: list[str] = []
        for _s in ordered_sources:
            _tk = group_doc_types.get(_s, "")
            _cap = _PER_TYPE_CAP.get(_tk, _PER_TYPE_CAP_DEFAULT)
            _n = type_counts.get(_tk, 0)
            if _n < _cap:
                type_counts[_tk] = _n + 1
                within_cap.append(_s)
            else:
                beyond_cap.append(_s)

        # Hide sources below relevance threshold; cap-overflow goes to hidden too.
        # If every within-cap source falls below the threshold, show them all
        # rather than presenting an empty result set -- the threshold only
        # filters when there is at least one source above it.
        # Web/doc types (web-content, user-guide) are exempt -- they are already
        # capped at their top N and should always be shown regardless of score.
        _THRESHOLD_EXEMPT = {"web-content", "user-guide"}
        above = [s for s in within_cap
                 if source_scores[s] >= _HIGH_RELEVANCE_THRESHOLD
                 or group_doc_types.get(s) in _THRESHOLD_EXEMPT]
        below = [s for s in within_cap
                 if source_scores[s] < _HIGH_RELEVANCE_THRESHOLD
                 and group_doc_types.get(s) not in _THRESHOLD_EXEMPT]
        if above:
            shown_sources = above
            hidden_sources = below + beyond_cap
        else:
            shown_sources = within_cap
            hidden_sources = beyond_cap

    # Render each source into a per-type bucket for grouped output.
    # Groups are emitted in the order their first member appears in shown_sources.
    _GROUP_LABEL: dict[str, str] = {
        "catalog-data": "Catalog Records",
        "schema": "Schema",
        "web-content": "Web Pages",
        "user-guide": "Documentation",
        "publication": "Publications",
    }
    seen_sentences: set[str] = set()
    rendered_count = 0
    # group_name -> list of rendered section strings
    group_buckets: dict[str, list[str]] = {}
    group_order: list[str] = []  # preserves insertion order for output
    for group_key in shown_sources:
        source = group_sources[group_key]
        score = source_scores[group_key]
        url = source_urls.get(group_key, "")

        # Derive a human-readable label and resolve a link URL.
        # Source key formats:
        #   "name:https://host/path/"  -- web crawled page (url in source key)
        #   "enriched:host:cid:schema:table"  -- dataset enricher (url in metadata)
        #   "name:path/to/file.md"  -- GitHub/local doc
        #   "path/to/file.md"  -- bare path (legacy/test)
        #   "name"  -- bare name
        #
        # Web-crawled chunks do not store a separate url metadata field; the
        # page URL is embedded in the source key as "name:https://...".
        if not url and ":" in source:
            _maybe_url = source.split(":", 1)[1]
            if _maybe_url.startswith("http://") or _maybe_url.startswith("https://"):
                url = _maybe_url

        from urllib.parse import urlparse as _up
        if url:
            _p = _up(url)
            # Chaise record URLs encode the useful info in the fragment
            # (e.g. #1/isa:dataset/RID=2B8P) rather than the path.
            if _p.fragment and "RID=" in _p.fragment:
                _rid = _p.fragment.split("RID=")[-1].split("/")[0].split("&")[0]
                label = f"RID={_rid}"
            elif _p.fragment and "/" in _p.fragment:
                _fsegs = [s for s in _p.fragment.split("/") if s]
                label = _fsegs[-1] if _fsegs else _p.netloc
            else:
                _segs = [s for s in _p.path.split("/") if s]
                label = _segs[-1] if _segs else _p.netloc
        elif source.startswith("enriched:"):
            # enriched:hostname:catalog_id:schema:table -> "schema: table"
            _parts = source.split(":")
            label = f"{_parts[-2]}: {_parts[-1]}" if len(_parts) >= 2 else source
        elif ":" in source:
            # name:path/to/file -> last non-empty path segment
            _path = source.split(":", 1)[1]
            _segs = [s for s in _path.split("/") if s]
            label = _segs[-1] if _segs else _path
        elif "/" in source:
            # bare path like "docs/guide.md" -> last segment
            _segs = [s for s in source.split("/") if s]
            label = _segs[-1] if _segs else source
        else:
            label = source

        entries = source_groups[group_key]
        # Prioritize the dataset title chunk (level-1 heading "# ...") so it
        # is processed first when scanning for the dataset title.
        entries = sorted(entries, key=lambda e: 0 if re.match(r"^# ", e.get("text", "")) else 1)

        is_enriched = source.startswith("enriched:")

        # Pre-scan entries to extract the dataset title (enriched records only).
        # Primary source: "title" metadata field set by the dataset enricher (any chunk).
        # Fallback: level-1 heading chunk "# Dataset RID\n\n{title}" (enriched only --
        # web pages also have level-1 headings but their body text is not a dataset title).
        dataset_title: str = ""
        title_chunk_matched: bool = False  # True when the level-1 title chunk was retrieved
        if is_enriched:
            # Try metadata first (populated after re-indexing with current enricher).
            for _e in entries:
                _mt = (_e.get("title") or "").strip()
                if _mt:
                    dataset_title = _mt
                    break
        # Level-1 heading check applies to all source types (sets title_chunk_matched);
        # body-text title extraction is enriched-only.
        for _e in entries:
            _t = _e.get("text", "")
            if re.match(r"^# ", _t):
                title_chunk_matched = True
                if is_enriched and not dataset_title:
                    _body = re.sub(r"^#[^\n]*\n+", "", _t).strip()
                    _first_line = next((ln.strip() for ln in _body.splitlines() if ln.strip()), "")
                    if _first_line:
                        dataset_title = _first_line
                break
        # Even when the title chunk was not retrieved, flag "title" as a matched
        # section if any significant query term appears verbatim in the title string.
        if is_enriched and not title_chunk_matched and dataset_title and question:
            _STOP = {"the", "a", "an", "and", "or", "of", "in", "for", "to", "with",
                     "is", "are", "was", "were", "this", "that"}
            _q_terms = {w for w in re.findall(r"[a-z]+", question.lower())
                        if len(w) >= 4 and w not in _STOP}
            _t_lower = dataset_title.lower()
            if _q_terms and any(term in _t_lower for term in _q_terms):
                title_chunk_matched = True

        _doc_type = (entries[0].get("doc_type", "") if entries else "") or (
            "catalog-data" if is_enriched else ""
        )

        # Metadata-only sections from the enricher that add no search context.
        _ENRICHED_SKIP = frozenset({
            "dataset identifiers", "contributors (authors)", "consortium",
            "study design",
        })
        # Navigation / UI text scraped from web pages.
        _NAV_RE = re.compile(
            r"^(view|read|download|next|prev|previous|home|back|skip|more|less"
            r"|login|sign\s+in|register)\b",
            re.IGNORECASE,
        )

        section_parts: list[str] = []
        matched_sections: list[str] = []  # section labels hit, for context line

        for entry in entries:
            text = entry.get("text", "")
            if not text:
                continue

            # Extract section label before stripping the heading.
            _hm = re.match(r"^#{1,3}\s*([^\n]+)\n+", text)
            section_label = _hm.group(1).strip() if _hm else ""

            # Strip heading and enricher title-context line.
            text = re.sub(r"^#{1,3}[^\n]*\n+", "", text).strip()
            text = re.sub(r"^Dataset:\s*[^\n]*\n+", "", text).strip()
            if not text:
                continue

            if is_enriched:
                if section_label.lower() in _ENRICHED_SKIP:
                    continue
                # Bullet-list vocab sections: "**Label:** item1, item2"
                bullet_items = [
                    re.sub(r"^\s*\*\s+", "", ln).strip()
                    for ln in text.splitlines()
                    if re.match(r"^\s*\*\s+", ln)
                ]
                bullet_items = [b for b in bullet_items if len(b) >= 2]
                if bullet_items:
                    items_str = ", ".join(bullet_items[:5])
                    part = f"**{section_label}:** {items_str}" if section_label else items_str
                    key = part.lower()
                    if key not in seen_sentences:
                        seen_sentences.add(key)
                        section_parts.append(part)
                        if section_label and section_label not in matched_sections:
                            matched_sections.append(section_label.lower())
                else:
                    # Free-text section: first sentence only.
                    for sent in _split_sentences(text):
                        sent = sent.strip()
                        if len(sent) < _MIN_SENTENCE_LEN:
                            continue
                        if dataset_title and sent == dataset_title:
                            continue
                        key = sent.lower()
                        if key not in seen_sentences:
                            seen_sentences.add(key)
                            section_parts.append(sent)
                            if section_label and section_label not in matched_sections:
                                matched_sections.append(section_label.lower())
                        break
            else:
                # Web / doc: show a paragraph-length excerpt from the highest-scoring chunk.
                # Drop very-short lines (single nav words like "Home", "All") and lines
                # matching NAV_RE ("View Publication", "Read more", etc.) before joining.
                # Threshold 8 preserves markdown table rows (14+ chars) while dropping
                # bare nav tokens. NAV_RE catches multi-word nav phrases regardless of length.
                _clean_lines = [
                    ln.strip() for ln in text.splitlines()
                    if len(ln.strip()) >= 8
                    and not _NAV_RE.search(ln.strip())
                ]
                _excerpt = " ".join(_clean_lines)
                if len(_excerpt) > 500:
                    _excerpt = _excerpt[:500].rsplit(" ", 1)[0] + "..."
                if _excerpt and _excerpt.lower() not in seen_sentences:
                    seen_sentences.add(_excerpt.lower())
                    section_parts.append(_excerpt)
                # Stop after first web/doc chunk -- one excerpt per source is enough
                break

        # Skip sources with no displayable content.
        if not dataset_title and not section_parts:
            continue

        rendered_count += 1

        # For enriched catalog records, flag "title" as a matched section when
        # the title chunk was retrieved -- it tells the user why the record ranked.
        # For web/doc sources, "title" is the display label itself and the match
        # was in the body, so the annotation would be redundant and misleading.
        if is_enriched and title_chunk_matched and "title" not in matched_sections:
            matched_sections.insert(0, "title")

        # Heading: for grouped output we drop the redundant _src_type from the
        # context line since the group header already conveys the source type.
        if matched_sections:
            _meta = f"matched in: {', '.join(matched_sections)} \u00b7 relevance: {score:.2f}"
        else:
            _meta = f"relevance: {score:.2f}"
        if url:
            heading = f"#### **[{label}]({url})** ({_meta})"
        else:
            heading = f"#### **{label}** ({_meta})"

        entry_lines: list[str] = [heading]

        # Dataset title always first for enriched records.
        if dataset_title:
            _dt = dataset_title if len(dataset_title) <= 120 else dataset_title[:117] + "..."
            entry_lines.append(f"**Title:** *{_dt}*")

        # Body: enriched up to 4 labeled sections; web/doc up to 3 sentences.
        cap = 4 if is_enriched else 3
        body = section_parts[:cap]
        if body:
            entry_lines.append("\n\n".join(body))

        # Route into the appropriate display group.
        _group_name = _GROUP_LABEL.get(_doc_type, "") or "Other"
        if _group_name not in group_buckets:
            group_buckets[_group_name] = []
            group_order.append(_group_name)
        group_buckets[_group_name].extend(entry_lines)
        group_buckets[_group_name].append("")  # blank line between entries

    # Emit each group with a header showing its count.
    parts.append(f"Results from {rendered_count} source(s):\n")
    for _gname in group_order:
        _glines = group_buckets[_gname]
        _gcount = sum(1 for l in _glines if l.startswith("####"))
        parts.append("---")
        parts.append(f"### {_gname} ({_gcount})")
        parts.extend(_glines)

    # Offer to show hidden lower-relevance sources
    hidden_count = len(hidden_sources)
    if hidden_count > 0 and not show_all:
        low_score = min(source_scores[s] for s in hidden_sources)
        high_score = max(source_scores[s] for s in hidden_sources)
        parts.append("---")
        parts.append(
            f"\n**{hidden_count} additional source(s) with lower relevance "
            f"({low_score:.2f}\u2013{high_score:.2f}) were not shown.** "
            "*Say \"show all results\" to include them.*"
        )

    return "\n".join(parts).rstrip()


# ---------------------------------------------------------------------------
# History trimming
# ---------------------------------------------------------------------------


def trim_history(messages: list[dict[str, Any]], max_turns: int) -> list[dict[str, Any]]:
    """Return messages trimmed to at most max_turns user exchanges.

    A "turn" starts with a role="user" message and includes all messages
    up to (but not including) the next role="user" message.  Tool messages
    (role="tool") and intermediate assistant messages are part of the same
    turn as the user message that triggered them.

    When trimming, always start on a user message boundary to avoid orphaning
    tool result messages whose associated assistant tool_calls were trimmed.
    """
    if not messages:
        return messages

    # Find indices of user messages (turn boundaries)
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]

    if len(user_indices) <= max_turns:
        return messages

    # Keep from the (len - max_turns)th user message onward
    start = user_indices[-max_turns]
    return messages[start:]


# ---------------------------------------------------------------------------
# Tool call accumulation from streaming deltas
# ---------------------------------------------------------------------------


def _accumulate_tool_call_deltas(
    acc: dict[int, dict[str, str]],
    delta_tool_calls: list[Any],
) -> None:
    """Merge streaming tool-call deltas into the accumulator.

    Each delta has an index identifying which tool call it belongs to.
    The first delta for a given index carries the id and function name;
    subsequent deltas carry argument fragments.
    """
    for tc_delta in delta_tool_calls:
        idx = tc_delta.index if hasattr(tc_delta, "index") else tc_delta.get("index", 0)
        if idx not in acc:
            acc[idx] = {"id": "", "name": "", "arguments": ""}

        # Extract id
        tc_id = getattr(tc_delta, "id", None) or (tc_delta.get("id") if isinstance(tc_delta, dict) else None)
        if tc_id:
            acc[idx]["id"] = tc_id

        # Extract function name and arguments
        func = getattr(tc_delta, "function", None) or (tc_delta.get("function") if isinstance(tc_delta, dict) else None)
        if func:
            fn_name = getattr(func, "name", None) or (func.get("name") if isinstance(func, dict) else None)
            fn_args = getattr(func, "arguments", None) or (func.get("arguments") if isinstance(func, dict) else None)
            if fn_name:
                acc[idx]["name"] = fn_name
            if fn_args:
                acc[idx]["arguments"] += fn_args


def _finalize_tool_calls(acc: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    """Convert the accumulator into a list of OpenAI tool_call dicts."""
    return [
        {
            "id": tc["id"],
            "type": "function",
            "function": {"name": tc["name"], "arguments": tc["arguments"]},
        }
        for _, tc in sorted(acc.items())
    ]


# ---------------------------------------------------------------------------
# Tool-calling loop
# ---------------------------------------------------------------------------


class ChatCancelled(Exception):
    """Raised when the client disconnects and the chat turn is aborted."""


def _user_label(session: "Session", include_email: bool = True) -> str:
    """Build a composite user identifier for LLM provider tracking and audit logs.

    Pass include_email=False when sending to LLM providers that reject email addresses
    in the user field (e.g. Anthropic).
    """
    cred = session.credenza_session or {}
    client_block = cred.get("client") or {}
    full_name = cred.get("full_name") or client_block.get("full_name") or ""
    email = cred.get("email") or client_block.get("email") or ""
    if include_email:
        if full_name or email:
            return f"{full_name} <{email}> ({session.user_id})"
    else:
        if full_name:
            return f"{full_name} ({session.user_id})"
    return session.user_id


async def run_chat_turn(
    user_message: str,
    session: Session,
    settings: Settings,
    cancelled: asyncio.Event | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run one chat turn and yield event dicts.

    Routes to the RAG-only response path when settings.operating_tier is
    'rag_only', otherwise runs the full LLM tool-calling loop.

    Yields:
      {"type": "status",      "message": str}              -- priming status update
      {"type": "text",        "content": str}              -- streamed text chunk
      {"type": "tool_start",  "name": str, "input": dict} -- before each tool call
      {"type": "tool_end",    "name": str, "result": str} -- after each tool call
      {"type": "turn_summary", ...}                        -- one per turn; consumed by server.py to emit chat_turn audit event; never forwarded to the SSE client

    Mutates session.history (appends the completed turn), session.tools
    (caches on first call), and session.schema_primed (set after first-turn
    priming).  The caller must persist the session after the iterator is
    exhausted.

    If *cancelled* is set, the generator stops at the next check point
    (before each LLM call and before each tool call) and saves partial
    history.

    Raises MCPAuthError if the MCP server rejects the bearer token.
    """
    # RAG-only mode: bypass the LLM loop entirely.
    # Applies when the server tier is rag_only, OR the user has toggled the
    # per-session override (only reachable when allow_rag_toggle is True), OR
    # the user is anonymous and rag_only_when_anonymous is enabled.
    is_anonymous = session.bearer_token is None
    if (settings.operating_tier == "rag_only"
            or session.rag_only_override
            or (settings.rag_only_when_anonymous and is_anonymous)):
        async for event in _rag_only_response(user_message, session, settings):
            yield event
        return

    mcp_url = settings.remap_url(settings.mcp_url)

    # First turn: open a single MCP session and batch all priming calls
    # (list_tools, guide prompts, schema, ERMrest syntax) on it to avoid
    # the per-call connection overhead (each connection does a full MCP
    # initialize handshake including ListToolsRequest).
    needs_tools = session.tools is None
    needs_priming = not session.schema_primed

    if needs_tools or needs_priming:
        yield {"type": "status", "message": "Connecting to server..."}
        async with open_session(session.bearer_token, mcp_url, ssl_verify=settings.ssl_verify) as mcp_sess:
            if needs_tools:
                session.tools = await list_tools(session.bearer_token, mcp_url, session=mcp_sess, ssl_verify=settings.ssl_verify)
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

    prompt = system_prompt(settings, session, schema_context, guide_context, ermrest_syntax)

    # Prompt caching: mark the system prompt and the tail of the tool list as
    # cacheable so that repeated calls within the cache TTL reuse the tokenized
    # blocks.  LiteLLM forwards cache_control to Anthropic (where it activates
    # prompt caching at 0.1x token cost); other providers silently ignore it.
    system_msg: dict[str, Any] = {
        "role": "system",
        "content": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
    }
    tools_with_cache: list[dict[str, Any]] = list(session.tools or [])
    if tools_with_cache:
        tools_with_cache[-1] = {**tools_with_cache[-1], "cache_control": {"type": "ephemeral"}}

    messages: list[dict[str, Any]] = list(session.history) + [
        {"role": "user", "content": user_message}
    ]

    def _check_cancelled() -> None:
        if cancelled is not None and cancelled.is_set():
            raise ChatCancelled()

    # Build model string: if llm_provider is set, prefix it (e.g. "ollama/llama3.1:8b")
    model = settings.llm_model
    if settings.llm_provider and "/" not in model:
        model = f"{settings.llm_provider}/{model}"

    # LiteLLM API kwargs that are constant across loop iterations
    api_kwargs: dict[str, Any] = {"api_key": settings.llm_api_key or None}
    if settings.llm_api_base:
        api_kwargs["api_base"] = settings.llm_api_base

    user_label = _user_label(session)                        # full composite for audit
    user_label_provider = _user_label(session, include_email=False)  # no email for LLM provider
    prev_tool_names: set[str] = set()  # tools called in the previous loop iteration

    # Accumulators for the per-turn audit summary.
    _audit_tools_invoked: list[str] = []
    _audit_tool_inputs: list[dict[str, Any]] = []
    _audit_tool_outputs: list[str] = []
    _audit_rag_triggered = False
    _audit_rag_docs: list[str] = []
    _audit_rag_scores: list[float] = []
    _audit_response_parts: list[str] = []

    while True:
        _check_cancelled()

        llm_messages = [system_msg] + messages

        response_content = ""
        tool_call_acc: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        captured_usage = None

        for attempt in range(_MAX_API_RETRIES + 1):
            text_yielded = False
            response_content = ""
            tool_call_acc = {}
            finish_reason = None
            captured_usage = None
            try:
                response_stream = await litellm.acompletion(
                    model=model,
                    messages=llm_messages,
                    tools=tools_with_cache if tools_with_cache else None,
                    max_tokens=_MAX_TOKENS,
                    stream=True,
                    stream_options={"include_usage": True},
                    user=user_label_provider,
                    **api_kwargs,
                )
                async for chunk in response_stream:
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        captured_usage = usage
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    if hasattr(delta, "content") and delta.content:
                        text_yielded = True
                        yield {"type": "text", "content": delta.content}
                        response_content += delta.content
                        if settings.audit_diagnostic:
                            _audit_response_parts.append(delta.content)
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        _accumulate_tool_call_deltas(tool_call_acc, delta.tool_calls)
                if captured_usage:
                    cache_read_tokens = getattr(captured_usage, "cache_read_input_tokens", 0) or 0
                    cache_creation_tokens = getattr(captured_usage, "cache_creation_input_tokens", 0) or 0
                    try:
                        prompt_cost, completion_cost = litellm.cost_per_token(
                            model=model,
                            prompt_tokens=captured_usage.prompt_tokens,
                            completion_tokens=captured_usage.completion_tokens,
                            cache_read_input_tokens=cache_read_tokens,
                            cache_creation_input_tokens=cache_creation_tokens,
                        )
                        cost = prompt_cost + completion_cost
                    except Exception:
                        cost = None
                    if cost is not None:
                        session.session_cost_usd += cost
                    session.session_prompt_tokens += captured_usage.prompt_tokens
                    session.session_completion_tokens += captured_usage.completion_tokens
                    session.session_cache_read_tokens += cache_read_tokens
                    session.session_cache_creation_tokens += cache_creation_tokens
                    audit_event(
                        "llm_usage",
                        user_id=user_label,
                        model=model,
                        prompt_tokens=captured_usage.prompt_tokens,
                        completion_tokens=captured_usage.completion_tokens,
                        cache_read_input_tokens=cache_read_tokens,
                        cache_creation_input_tokens=cache_creation_tokens,
                        cost_usd=cost,
                    )
                break  # success -- exit retry loop
            except litellm.RateLimitError:
                if text_yielded or attempt >= _MAX_API_RETRIES:
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.0fs",
                    attempt + 1, _MAX_API_RETRIES, delay,
                )
                await asyncio.sleep(delay)
            except litellm.ServiceUnavailableError:
                if text_yielded or attempt >= _MAX_API_RETRIES:
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "API overloaded (attempt %d/%d), retrying in %.0fs",
                    attempt + 1, _MAX_API_RETRIES, delay,
                )
                await asyncio.sleep(delay)

        # Build assistant message for history
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if response_content:
            assistant_msg["content"] = response_content
        else:
            assistant_msg["content"] = None

        tool_calls = _finalize_tool_calls(tool_call_acc) if tool_call_acc else []
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        if finish_reason != "tool_calls" or not tool_calls:
            # End of turn -- no tools to execute
            messages.append(assistant_msg)
            break

        # Execute tool calls
        curr_tool_names = {tc["function"]["name"] for tc in tool_calls}
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

        messages.append(assistant_msg)

        for tc in tool_calls:
            tc_id = tc["id"]
            tc_name = tc["function"]["name"]
            tc_args_str = tc["function"]["arguments"]

            try:
                tc_args = json.loads(tc_args_str) if tc_args_str else {}
            except json.JSONDecodeError:
                tc_args = {}

            _check_cancelled()
            yield {"type": "tool_start", "name": tc_name, "input": tc_args}
            _audit_tools_invoked.append(tc_name)
            try:
                result_text = await call_tool(
                    session.bearer_token, tc_name, tc_args, mcp_url, ssl_verify=settings.ssl_verify
                )
            except MCPAuthError:
                raise
            except Exception as exc:
                logger.error("Tool %s failed: %s", tc_name, exc)
                result_text = f"Error executing tool {tc_name}: {exc}"
            yield {"type": "tool_end", "name": tc_name, "result": result_text[:_TOOL_RESULT_PREVIEW]}
            if tc_name == "rag_search":
                _audit_rag_triggered = True
                try:
                    _rag_hits: list[dict[str, Any]] = json.loads(result_text)
                    for _hit in _rag_hits:
                        _audit_rag_docs.append(_hit.get("url") or _hit.get("source", ""))
                        _audit_rag_scores.append(float(_hit.get("score", 0)))
                except Exception:
                    pass  # best-effort; malformed result leaves accumulators as-is
            if settings.audit_diagnostic:
                _audit_tool_inputs.append({"name": tc_name, "args": tc_args})
                _out_max = settings.audit_diagnostic_tool_output_max_chars
                _audit_tool_outputs.append(result_text[:_out_max] if _out_max > 0 else result_text)

            llm_content = result_text[:_TOOL_RESULT_TO_LLM]
            if len(result_text) > _TOOL_RESULT_TO_LLM:
                llm_content += "\n[result truncated]"
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": llm_content})

    truncated = _truncate_history_tool_results(messages)
    session.history = trim_history(truncated, settings.max_history_turns)

    # Yield the per-turn audit summary for the server layer to emit.
    summary: dict[str, Any] = {
        "tools_invoked": _audit_tools_invoked,
        "rag_triggered": _audit_rag_triggered,
        "rag_document_count": len(_audit_rag_docs),
        "rag_documents": _audit_rag_docs,
        "rag_scores": _audit_rag_scores,
        "model": model,
    }
    if settings.audit_diagnostic:
        _resp_max = settings.audit_diagnostic_response_max_chars
        full_response = "".join(_audit_response_parts)
        summary["user_query"] = user_message
        summary["response_text"] = full_response[:_resp_max] if _resp_max > 0 else full_response
        summary["response_compressed"] = base64.b64encode(gzip.compress(full_response.encode())).decode()
        summary["tool_inputs"] = _audit_tool_inputs
        summary["tool_outputs"] = _audit_tool_outputs
    yield {"type": "turn_summary", **summary}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_history_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return messages with tool result content truncated to _HISTORY_TOOL_RESULT_MAX chars.

    The full result is used within the current turn (fed back to the LLM), but
    replaying large tool outputs in every subsequent request wastes input tokens.
    This truncation only affects the stored history copy, not the live messages list.
    """
    out = []
    for msg in messages:
        if (
            msg.get("role") == "tool"
            and isinstance(msg.get("content"), str)
            and len(msg["content"]) > _HISTORY_TOOL_RESULT_MAX
        ):
            out.append({**msg, "content": msg["content"][:_HISTORY_TOOL_RESULT_MAX] + "\n[truncated]"})
        else:
            out.append(msg)
    return out
