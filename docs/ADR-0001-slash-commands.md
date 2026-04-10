# ADR-0001: Slash Commands for Tools, Resources, and Prompts

**Status:** Proposed  
**Date:** 2026-04-09  
**Context:** deriva-mcp-ui chatbot UI

---

## Context

The chatbot UI currently requires users to express all intent in natural language, which the LLM then interprets to decide which MCP tools to call. For common, well-defined operations (search, schema inspection, re-indexing) this adds latency and non-determinism. Power users benefit from a direct shortcut syntax.

The chatbot operates in two modes (`operating_mode`):
- **llm** -- LLM-mediated tool calls; natural language in, structured output out
- **rag_only** -- no LLM; RAG search results formatted by Python code directly

Slash commands should behave differently in each mode rather than being mode-blind.

---

## Decision

Implement a `/command [args]` syntax in the chat textarea with the following properties:

### 1. Mode-sensitive dispatch

| Mode | Behaviour |
|------|-----------|
| `llm` | Command is rewritten to a terse directive and sent through the normal LLM loop. The LLM executes the tool call immediately and formats the result per existing system prompt rules (numbered lists, links, summaries). |
| `rag_only` | Command is dispatched directly to the MCP tool via `call_tool()`, bypassing the LLM entirely. Result is formatted by Python code. |

This keeps client-side logic mode-agnostic: the client always sends the raw `/command args` string to `POST /chat`. The server detects the slash prefix and routes accordingly.

### 2. Command registry

A static curated registry maps short slash-command names to MCP tool names and argument signatures. Descriptions are populated at request time from the live MCP `tools/list` response so they stay accurate as the MCP server evolves.

**Initial registry (Phase 1):**

| Command | Tool | Args |
|---------|------|------|
| `/search` | `rag_search` | `<query>` |
| `/status` | `rag_status` | _(none)_ |
| `/ingest` | `rag_ingest` | `<source-name>` |
| `/schema` | `get_schema` | _(none)_ |
| `/tasks` | `list_tasks` | _(none)_ |
| `/clear` | _(client-side)_ | _(none)_ |
| `/help` | _(client-side)_ | _(none)_ |

**Excluded from Phase 1:**
- Flags/options (e.g. `--force`) -- deferred
- MCP prompt invocation (e.g. `/analyze-dataset`) -- deferred
- MCP resource access (e.g. `/resource deriva://...`) -- deferred

### 3. `GET /commands` endpoint

Returns the enriched command list. Called lazily on the first `/` keystroke of a session (not at page load), so a valid session (anonymous or authenticated) is guaranteed to exist. Result is cached client-side for the remainder of the session.

**Request:** `GET /commands` (requires session cookie, works for anonymous sessions)

**Response:**
```json
[
  {
    "name": "search",
    "arg_hint": "<query>",
    "tool": "rag_search",
    "description": "<from tools/list>",
    "client_only": false
  },
  {
    "name": "clear",
    "arg_hint": "",
    "tool": null,
    "description": "Clear conversation history",
    "client_only": true
  }
]
```

The server calls `list_tools(session.bearer_token, settings.mcp_url)` to fetch live descriptions. `bearer_token` may be `None` for anonymous sessions; `list_tools` already handles this.

### 4. Autocomplete dropdown

- Triggered on `/` as the first character on a line in the textarea
- Filters dynamically as the user continues typing
- Keyboard navigation: Arrow keys, Tab/Enter to complete, Escape to dismiss
- Shows command name, arg hint, and description for each entry
- Dismissed automatically when input no longer starts with `/`

### 5. Client-side commands

`/clear` and `/help` are intercepted in JavaScript before the message reaches the server:

- `/clear` -- calls `DELETE /history` directly (same as the existing Clear button), resets local command history state
- `/help` -- renders the command list inline in the chat pane from the cached registry; no server round-trip

### 6. LLM directive injection (llm mode)

When the server detects a `/command [args]` prefix in llm mode, it rewrites the user message before it enters `run_chat_turn`:

```
/search cleft palate
→
"[Direct command: call rag_search with query='cleft palate'. Execute immediately without deliberation or clarifying questions.]"
```

The original `/command args` string is stored in history as the user turn (not the rewritten directive) so the conversation history remains readable.

### 7. RAG-only direct dispatch

In `rag_only` mode the server parses the command, maps it to a tool name and argument dict, calls `call_tool()` directly, and streams the result via the existing SSE format. The formatting layer reuses the existing `_format_rag_response` for `/search`, and a simple JSON-to-Markdown formatter for other tools.

---

## Command argument parsing (Phase 1)

Simple: everything after the command name (trimmed) is the argument string. No flag parsing.

```
/search cleft palate AND mouse     → query = "cleft palate AND mouse"
/ingest facebase-web               → source_name = "facebase-web"
/schema                            → no args
```

Multi-word tool arguments that require structured input (e.g. hostname + catalog_id for `/schema`) use the session's default catalog context where possible, falling back to a clarifying prompt if context is missing.

---

## Deferred / Future Enhancements

### Phase 2: Flags
```
/ingest facebase-web --force
/search cleft palate --limit 20 --doc-type web-content
```
Requires a lightweight flag parser (no external deps; hand-rolled is fine given the small surface).

### Phase 3: Prompt invocation
```
/analyze-dataset RID=2B8P
/explain-schema isa:dataset
```
MCP `prompts/list` is fetched alongside `tools/list` at the `/commands` endpoint. Prompt arguments use `key=value` syntax. The server calls `get_prompt(name, args)` and injects the result as a system context block or user turn.

### Phase 4: Resource access
```
/resource deriva://staging.facebase.org/catalog/1/schema
```
Fetches an MCP resource URI and renders the content inline. Useful for inspecting live schema or metadata without a full tool call round-trip.

### Phase 5: Command aliases and user-defined shortcuts
Allow users to define aliases in session preferences (e.g. `/fb` → `/search facebase`). Stored in session or browser localStorage.

---

## Implementation Checklist

### Server (deriva-mcp-ui)
- [ ] `GET /commands` endpoint in `server.py` -- calls `list_tools`, merges with static registry, returns JSON
- [ ] Slash command detection in `POST /chat` handler -- regex on incoming message
- [ ] LLM directive rewriter -- `_rewrite_slash_command(cmd, args, session, settings) -> str`
- [ ] RAG-only dispatcher -- `_dispatch_slash_command(cmd, args, session, settings)` returning SSE generator
- [ ] History storage: store original `/command args` as user turn, not the rewritten directive
- [ ] Tests: command detection, rewrite, RAG-only dispatch, `/commands` endpoint

### Client (chat.js)
- [ ] Lazy fetch of `GET /commands` on first `/` keystroke; cache result
- [ ] Autocomplete dropdown component (pure JS/CSS, no framework)
- [ ] Client-side `/clear` and `/help` interception
- [ ] Textarea keydown handler: trigger autocomplete on `/` at start of line
- [ ] Send raw `/command args` to `POST /chat` unchanged (server handles rewrite)
