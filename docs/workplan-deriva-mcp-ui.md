# deriva-mcp-ui Workplan and Design

**Status:** Phases 0-8 complete + post-Phase-8 anonymous mode hardening + schema priming/guide injection (Phase 9 design
only)

**Target repo:** `deriva-mcp-ui`

---

## Overview

`deriva-mcp-ui` is a browser-based chatbot interface for DERIVA. It wraps the Claude LLM
and the `deriva-mcp-core` MCP server behind a web frontend, giving end users a natural
language interface to query and manage DERIVA catalogs without needing a desktop MCP client
(Claude Desktop, Claude Code, etc.).

Design goals:

- Standard web login via Credenza (no token pasting, no special client setup)
- Two operating modes: **default-catalog** (anchored to a specific catalog, for production
  deployments) and **general-purpose** (user specifies catalog, for platform-level access)
- Streaming responses via SSE -- Claude's replies appear incrementally
- Clean separation from `deriva-mcp-core` -- the UI service is an MCP client over HTTP;
  the MCP server is unchanged
- Deployable in the `deriva-docker` compose stack or behind Apache in a VM deployment

---

## Architecture

```
Browser (HTML + JS)
  |
  | HTTPS (session cookie)
  v
deriva-mcp-ui  (FastAPI/Starlette)
  |                    |
  | MCP HTTP           | HTTPS
  | (bearer token)     | (Claude API)
  v                    v
deriva-mcp-core     Anthropic API
  |
  | HTTPS
  v
DERIVA (ERMrest, Hatrac)
  |
  v
Credenza (OAuth AS -- introspect + exchange, as today)
```

### What the UI service does

1. Serves the browser chat UI (static HTML/JS)
2. Handles Credenza OAuth login and manages server-side user sessions
3. On each chat request: connects to the MCP server as an MCP client using the user's
   bearer token, runs the Claude tool-calling loop, streams text chunks back to the browser
   via SSE

### What the UI service does NOT do

- Execute DERIVA tools directly -- all tool execution happens inside `deriva-mcp-core`
- Validate or exchange tokens itself -- the MCP server's auth layer handles that
- Maintain a persistent MCP connection -- each chat turn opens a fresh stateless-HTTP
  MCP request (matching the MCP server's `stateless_http=True` model)

---

## Token Chain

The token chain has three layers. Understanding which layer controls which lifetime is
important for configuring the deployment correctly.

```
Credenza issues user bearer token
  (lifetime: operator-configured on the UI client registration -- e.g. 8h, 16h, 24h)
      |
      | UI service presents this token to the MCP server
      v
MCP server introspects via Credenza, exchanges for a derived DERIVA token
  (lifetime: 30-min hard cap on SessionType.DERIVED -- managed by DerivedTokenCache)
      |
      | Derived token used for all ERMrest / Hatrac calls within the request
      v
DERIVA
```

The 30-minute derived token cap is transparent to the UI layer. `DerivedTokenCache` in
`deriva-mcp-core` re-exchanges automatically on every cache miss as long as the upstream
bearer token is still valid. The UI service only ever holds and presents the user's
long-lived bearer token.

The only session lifecycle event the UI service must handle is the user's Credenza session
expiry (at whatever TTL is configured on the UI client registration). When the bearer token
expires, the MCP server returns 401, and the UI service redirects the browser to re-authenticate.

---

## Auth Flow

The UI service acts as a Credenza Relying Party (same model as Chaise) using the
authorization_code flow with PKCE.

```
1. Browser hits /
2. No valid session -> redirect to Credenza GET /authorize
      client_id=deriva-mcp-ui
      response_type=code
      resource=<DERIVA_MCP_SERVER_RESOURCE>   <-- request token scoped for MCP server
      redirect_uri=<DERIVA_CHATBOT_PUBLIC_URL>/callback
      code_challenge=<PKCE challenge>
      state=<CSRF token>

3. User authenticates at Credenza (Keycloak, Globus, etc.)
4. Credenza redirects to /callback?code=...&state=...
5. UI service: verify state, POST /token (authorization_code grant) with PKCE verifier
6. Credenza returns bearer token scoped to DERIVA_MCP_SERVER_RESOURCE
7. UI service stores token in server-side session (keyed by secure session cookie)
8. Browser is redirected to the chat UI

On each chat request:
9. UI service reads bearer token from session
10. Connects to MCP server with Authorization: Bearer <token>
11. MCP server validates via Credenza introspect + exchange as normal
```

By requesting `resource=<DERIVA_MCP_SERVER_RESOURCE>` in step 2, the token Credenza
issues is already scoped for the MCP server. No second token exchange step is needed at
the UI layer.

### Mutation access

The `DERIVA_MCP_MUTATION_REQUIRED_CLAIM` on the MCP server provides natural mutation
gating for UI users. If configured as `{"groups": ["deriva-mcp-mutator"]}`, only users
whose Credenza token carries that group claim can execute mutating tools -- whether they
connect via Claude Desktop, Claude Code, or this UI. No special UI-side logic required.

---

## Operating Modes

### Default-catalog mode

Activated when `DERIVA_CHATBOT_DEFAULT_HOSTNAME` and `DERIVA_CHATBOT_DEFAULT_CATALOG_ID`
are both set.

- The chat UI hides the hostname/catalog input fields
- The system prompt anchors Claude to the specific catalog:

```
You are a DERIVA data assistant for the <LABEL> catalog. You have access to tools
for querying and managing this catalog. When answering questions about data, schema,
or annotations, use the available tools rather than relying on prior knowledge.
```

- At the start of each new conversation, the UI service calls `get_catalog_info` and
  `get_schema` to prime Claude's context with schema information, then injects MCP guide
  prompts and ERMrest syntax documentation into the system prompt before the user's first
  message is processed. Primed content is cached in the session for subsequent turns.

### General-purpose mode

Activated when the default catalog vars are unset.

- The chat UI shows hostname and catalog ID input fields (optional -- Claude can also
  ask the user for them in the conversation)
- The system prompt is broader:

```
You are a DERIVA data assistant. You have access to tools for querying and managing
DERIVA catalogs. When the user wants to work with a specific catalog, ask for the
hostname and catalog ID if they have not been provided.
```

---

## Claude Tool-Calling Loop

The chat endpoint runs the full tool-calling loop server-side and streams text chunks
to the browser via SSE. Tool invocations are not shown in the browser by default
(configurable to show "Searching..." indicators).

```python
# Pseudocode
async def chat_turn(user_message, session):
    async with mcp_client(session.bearer_token) as mcp:
        if session.tools is None:
            session.tools = to_anthropic_schema(await mcp.list_tools())

        messages = session.history + [{"role": "user", "content": user_message}]

        while True:
            async with claude.messages.stream(
                    model=settings.claude_model,
                    system=system_prompt(session),
                    tools=session.tools,
                    messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield sse_text(text)  # -> browser

                response = await stream.get_final_message()

            if response.stop_reason == "end_turn":
                break

            # execute tool_use blocks via MCP
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await mcp.call_tool(block.name, block.input)
                    tool_results.append(tool_result_block(block.id, result))

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        session.history = trim_history(messages)
```

### Tool schema conversion

MCP tool definitions include `name`, `description`, and a JSON schema for `inputSchema`.
The Anthropic API accepts `name`, `description`, and `input_schema` -- the same fields,
different key name. Conversion is a shallow rename plus passthrough of the schema object.

---

## Conversation History

History is stored server-side, keyed by the session ID (from the session cookie). In-memory
for a single-instance deployment; a Redis backend is provided for multi-instance.

History is trimmed when it approaches Claude's context limit. Two strategies (configurable):

- **Turn-count trim**: keep the last N turns (simple, predictable)
- **Token-count trim**: keep as many recent turns as fit within a token budget (more
  accurate, requires a token counter)

The system prompt is not counted as history and is always prepended fresh.

---

## Configuration

`pydantic-settings` `BaseSettings` with `DERIVA_CHATBOT_` prefix.

**Required:**

| Variable                        | Description                                                           |
|---------------------------------|-----------------------------------------------------------------------|
| `DERIVA_CHATBOT_MCP_URL`        | Base URL of the deriva-mcp-core server                                |
| `DERIVA_CHATBOT_CREDENZA_URL`   | Base URL of the Credenza instance                                     |
| `DERIVA_CHATBOT_CLIENT_ID`      | OAuth client ID registered in Credenza for this service               |
| `DERIVA_CHATBOT_CLIENT_SECRET`  | OAuth client secret                                                   |
| `DERIVA_CHATBOT_MCP_RESOURCE`   | Resource identifier for the MCP server (must match MCP server config) |
| `DERIVA_CHATBOT_PUBLIC_URL`     | Public HTTPS URL of this service (used as OAuth redirect base)        |
| `ANTHROPIC_API_KEY`             | Anthropic API key for Claude                                          |
| `DERIVA_CHATBOT_SESSION_SECRET` | Secret key for signing session cookies (random bytes, keep private)   |

**Default-catalog mode (both required to activate):**

| Variable                               | Description                                    |
|----------------------------------------|------------------------------------------------|
| `DERIVA_CHATBOT_DEFAULT_HOSTNAME`      | DERIVA server hostname for the default catalog |
| `DERIVA_CHATBOT_DEFAULT_CATALOG_ID`    | Catalog ID or alias                            |
| `DERIVA_CHATBOT_DEFAULT_CATALOG_LABEL` | Display name shown in the UI (optional)        |

**Tuning:**

| Variable                             | Default                   | Description                                                                                                                                                              |
|--------------------------------------|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DERIVA_CHATBOT_CLAUDE_MODEL`        | `claude-haiku-4-5-latest` | Claude model ID (Haiku is sufficient after prompt engineering work; override to Sonnet if needed)                                                                        |
| `DERIVA_CHATBOT_MAX_HISTORY_TURNS`   | `10`                      | Max conversation turns retained in server-side history                                                                                                                   |
| `DERIVA_CHATBOT_SESSION_TTL`         | `28800`                   | Server-side session TTL in seconds (default 8h)                                                                                                                          |
| `DERIVA_CHATBOT_STORAGE_BACKEND`     | `memory`                  | Session store backend: `memory`, `redis`, `valkey`, `postgresql`, `sqlite`                                                                                               |
| `DERIVA_CHATBOT_STORAGE_BACKEND_URL` | --                        | Connection URL for the selected backend (not used for `memory`). Examples: `redis://localhost:6379/0`, `postgresql://user:pass@host/db`, `sqlite:///path/to/sessions.db` |
| `DERIVA_CHATBOT_DEBUG`               | `false`                   | Enable debug logging and show tool calls in the UI                                                                                                                       |

---

## Package Structure

```
deriva-mcp-ui/
├── pyproject.toml
├── Dockerfile
├── docs/
│   └── workplan-deriva-mcp-ui.md
└── src/
    └── deriva_mcp_ui/
        ├── __init__.py
        ├── server.py        # FastAPI app, route registration, lifespan
        ├── config.py        # Settings (DERIVA_CHATBOT_* vars)
        ├── auth.py          # Credenza OAuth client: /login, /callback, /logout routes
        ├── storage/         # Session store backends (mirrors Credenza storage pattern)
        │   ├── __init__.py  # STORAGE_BACKENDS registry + factory
        │   ├── base.py      # SessionStore protocol + Session dataclass
        │   ├── memory.py
        │   ├── redis.py
        │   ├── valkey.py
        │   ├── postgresql.py
        │   └── sqlite.py
        ├── mcp_client.py    # MCP client wrapper: connect, list_tools, call_tool
        ├── chat.py          # Claude tool-calling loop + SSE response streaming
        └── static/
            ├── index.html   # Chat UI shell
            └── chat.js      # SSE client, message rendering, login state
```

---

## Deployment

### Docker compose

Add a `deriva-mcp-ui` service to the `deriva-docker` compose stack. The service connects
to `deriva-mcp-core` over the internal Docker network (e.g., `http://deriva-mcp-core:8000`)
and to Credenza over the same network or via the public URL (depending on network topology).

Traefik routes:

- `/chatbot/` -> `deriva-mcp-ui:8001`

The MCP URL from the UI service's perspective is the internal network address. The public
URL (`DERIVA_CHATBOT_PUBLIC_URL`) is used only for the OAuth redirect URI.

### Apache (VM deployment)

```apache
ProxyPass /chatbot/ http://127.0.0.1:8001/
ProxyPassReverse /chatbot/ http://127.0.0.1:8001/
```

The UI service listens on `127.0.0.1:8001`. TLS termination is handled by Apache as with
all other services in the stack.

### Credenza client registration

A new client entry is required in `oidc_clients.json` (or equivalent Credenza config):

```json
{
  "client_id": "deriva-mcp-ui",
  "client_secret": "<secret>",
  "grant_types": [
    "authorization_code"
  ],
  "redirect_uris": [
    "https://example.org/chatbot/callback"
  ],
  "allowed_resources": [
    "<DERIVA_MCP_SERVER_RESOURCE>"
  ],
  "token_ttl": 86400
}
```

`token_ttl` controls the user bearer token lifetime. Set to a value appropriate for the
deployment (e.g., 28800 for 8h, 86400 for 24h). Users whose session expires see a
re-authentication redirect -- no data loss, the conversation history on the server side
can optionally be preserved across re-auth if keyed by a stable user identifier.

---

## Phases

---

### Phase 0 -- Scaffolding

- Create `deriva-mcp-ui` repo (or work in `deriva-mcp-core` until ready to move)
- `pyproject.toml` with `uv` / `hatchling`; dependencies: `fastapi`, `httpx`,
  `anthropic`, `mcp`, `pydantic-settings`, `itsdangerous` (session signing)
- `src/deriva_mcp_ui/` package layout
- `ruff` for lint/format, `pytest` + `pytest-asyncio` for tests
- `Dockerfile`: slim Python image, `uv sync`, `uvicorn` entrypoint
- CI scaffold

Deliverable: `uv sync` and `pytest` run cleanly (zero tests, no errors).

---

### Phase 1 -- Configuration and Auth

**1.1 Config**

`config.py`: `Settings(BaseSettings)` with all `DERIVA_CHATBOT_*` variables.
`validate()`: raise on missing required fields; also validate that if either default
catalog var is set, both are set.

**1.2 Credenza OAuth client**

`auth.py`:

- `GET /login` -- build Credenza `/authorize` URL with PKCE challenge and `state` CSRF
  token, set a short-lived cookie with the PKCE verifier and state, redirect to Credenza
- `GET /callback` -- verify state, POST to Credenza `/token` with PKCE verifier, store
  bearer token in server-side session, redirect to `/`
- `GET /logout` -- clear session, redirect to Credenza `/logout` (if supported) or to `/`
- `_require_session` dependency -- used on protected routes; redirects to `/login` if no
  valid session

**1.3 Session store**

`storage/` subpackage, mirroring the Credenza storage backend pattern:

```
storage/
  __init__.py      # STORAGE_BACKENDS registry + factory(backend, url) -> SessionStore
  base.py          # SessionStore protocol: get / set / delete / sweep
  memory.py        # In-process dict + TTL sweep (default, dev/single-instance)
  redis.py         # Redis backend (recommended for multi-instance)
  valkey.py        # Valkey backend (drop-in Redis-compatible alternative)
  postgresql.py    # PostgreSQL backend (persistent; survives restarts)
  sqlite.py        # SQLite backend (lightweight persistent; single-instance VM)
```

`Session` dataclass (defined in `base.py`): `user_id`, `bearer_token`, `history`,
`tools` (cached tool list), `created_at`, `last_active`.

The `SessionStore` protocol exposes `get(session_id)`, `set(session_id, session)`,
`delete(session_id)`, and `sweep()` (evict expired entries). All backends serialize
`Session` to JSON. The active backend is selected at startup from
`DERIVA_CHATBOT_STORAGE_BACKEND` + `DERIVA_CHATBOT_STORAGE_BACKEND_URL` via the factory.

Session ID stored in a signed `HttpOnly Secure SameSite=Lax` cookie via `itsdangerous`.

Backend guidance (same as Credenza):

- `memory` -- development and single-worker deployments only; state lost on restart
- `redis` / `valkey` -- recommended for production multi-instance deployments
- `postgresql` -- persistent; use when a PostgreSQL instance is already in the stack
- `sqlite` -- lightweight persistent option for single-instance VM deployments without Redis

Deliverable: `/login` and `/callback` complete the OAuth flow and set a session cookie.
`/logout` clears it. Protected routes redirect correctly without a session.

---

### Phase 2 -- MCP Client

`mcp_client.py`:

- `list_tools(bearer_token) -> list[AnthropicTool]` -- connect to MCP server, call
  `session.list_tools()`, convert to Anthropic tool schema format (rename `inputSchema`
  to `input_schema`, passthrough everything else), return list
- `call_tool(bearer_token, name, arguments) -> str` -- connect, call
  `session.call_tool(name, arguments)`, extract text content from result, return as string
- Both functions open a fresh `streamablehttp_client` connection per call by default
  (stateless HTTP model; no persistent connection to maintain)
- `open_session(bearer_token)` async context manager provides a shared `ClientSession`
  for batching multiple MCP calls on a single connection (used during schema priming to
  avoid per-call `ListToolsRequest` overhead)
- Connection errors (MCP server unreachable, 401 from MCP server) raise typed exceptions
  that the chat layer converts to user-visible error messages

Deliverable: unit tests with a mocked MCP server confirming tool listing and tool call
round-trips.

---

### Phase 3 -- Claude Integration and Chat Endpoint

`chat.py`:

- `system_prompt(session) -> str` -- returns catalog-anchored prompt or general prompt
  depending on config; if default-catalog mode, primes schema context via `rag_search`
  at the start of the first turn (result appended to system prompt, not shown to user)
- `run_chat_turn(user_message, session, bearer_token) -> AsyncIterator[str]` -- the
  full tool-calling loop; yields SSE-formatted text chunks as Claude generates them;
  calls `mcp_client.call_tool()` for each `tool_use` block; appends completed turn to
  `session.history`; trims history at `MAX_HISTORY_TURNS`
- `trim_history(messages, max_turns) -> list` -- keep system messages untouched, trim
  oldest user/assistant pairs

`server.py`:

- `GET /` -- serve `static/index.html` (redirects to `/login` if no session)
- `POST /chat` -- accepts `{"message": "..."}` JSON body; calls `run_chat_turn`;
  returns `text/event-stream` SSE response streaming text chunks; final event is
  `event: done`
- `GET /session-info` -- returns `{"user_id": ..., "catalog_mode": ..., "label": ...}` for
  the JS to render in the UI header

Deliverable: end-to-end chat against a live (or mocked) MCP server, streamed to a
`curl` or `httpie` client.

---

### Phase 4 -- Browser UI

`static/index.html` and `static/chat.js`:

- Clean, minimal chat layout: message thread, input box, send button
- SSE client using `EventSource` or `fetch` with streaming body (prefer `fetch` for
  POST support; `EventSource` only supports GET)
- Message rendering: user messages right-aligned, assistant messages left-aligned with
  Markdown rendering (use a lightweight library like `marked.js` -- no framework needed)
- Login state: if the server returns 401 on `/chat`, redirect to `/login`
- Loading indicator while Claude is thinking (before first SSE chunk arrives)
- Catalog label in the header (populated from `/session-info`)
- In general-purpose mode: hostname and catalog ID fields shown above the input box;
  values passed in the `POST /chat` body and stored in the session

No build step. Plain HTML + vanilla JS + one CDN script for Markdown rendering.

Deliverable: functional browser UI against a running stack.

---

### Phase 5 -- Default Catalog Schema Priming

**Status:** Complete (reimplemented 2026-04-03).

When default-catalog mode is active, the first turn of a new conversation primes the
system prompt with schema context, MCP guide prompts, and ERMrest syntax documentation.

**Schema priming** (`_prime_schema`): calls `get_catalog_info` to discover schemas, then
`get_schema` for each schema (skipping `public` which contains ERMrest system tables).
Results are concatenated under a 20k-character budget; the first schema is always included
even if it exceeds the budget. The original design used `rag_search`, but testing showed
that structured `get_schema` output provides cleaner, more complete schema context than
RAG fragments.

**Guide injection** (`_fetch_guides`): fetches `query_guide`, `entity_guide`, and
`annotation_guide` MCP prompts and injects them into the system prompt. These provide
behavioral directives (mandatory rules, anti-pattern prevention) that are more effective
in the system prompt than in tool descriptions, especially with Haiku.

**ERMrest syntax** (`_prime_ermrest_syntax`): fetches ERMrest URL syntax documentation via
`rag_search` for additional query reference material.

**Connection batching** (`open_session`): all priming calls share a single MCP session
via `mcp_client.open_session()`, avoiding the per-call `ListToolsRequest` overhead.

**Persistence**: primed content is stored on the session (`primed_schema`, `primed_guides`,
`primed_ermrest`) and reused on subsequent turns without re-fetching. Cleared on
conversation reset.

Deliverable: in default-catalog mode, Claude answers schema questions correctly on the
first turn without the user needing to say "look at the schema first."

---

### Phase 6 -- Deployment Wiring

- `Dockerfile`: multi-stage build; final image runs `uvicorn deriva_mcp_ui.server:app`
- `deriva-docker` compose service entry with env var passthrough
- Traefik label config for `/chatbot/` path routing
- Apache `ProxyPass` snippet for VM deployments (documented, not automated)
- Credenza client registration documented with all required fields and recommended TTL values
- `README.md`: quickstart, configuration reference, deployment options

Deliverable: running in the `deriva-docker` compose stack with a real Credenza login.

---

### Phase 7 -- Hardening

Implemented:

- **Session architecture**: two-key store pattern preserves conversation history across
  re-authentication (see design note below)
- **Input length limit**: `DERIVA_CHATBOT_MAX_MESSAGE_LENGTH` (default 10,000 chars);
  `/chat` returns 400 if exceeded
- **Auth UX**: `/session-info` returns full `credenza_session` dict; JS handles
  logged-out state without server-side redirects; logout clears page state
- **`parsed_output` fix**: `_content_to_dicts` uses a field whitelist (`_BLOCK_FIELDS`)
  to strip SDK-internal fields from Anthropic content blocks before storing in history,
  preventing API validation errors on history replay
- **SSE hang fix**: outer `while` reader loop now uses a `streamDone` flag so the
  `done` event correctly exits the loop (inner `break` was insufficient)
- **Tool call visibility**: tool events streamed as `event: tool` SSE; JS renders
  collapsible `<details>` blocks with input and result sections
- **Numbered list instruction**: system prompt instructs Claude to always use numbered
  lists for options, enabling single-digit replies

Not implemented (deferred):

- Rate limiting per session
- Structured JSON logging (plain stdlib logging is sufficient for v1)
- `/health` endpoint -- already present from Phase 3

---

### Session Architecture Note

**Decision:** conversation history is preserved across re-authentication.

When a Credenza bearer token expires the user re-authenticates and gets a new token.
Rather than starting a fresh conversation, the server links the new token to the
existing session via a two-key pattern in the session store:

```
tok:{bearer_token}  ->  minimal Session (user_id only)     TTL = session_ttl
uid:{user_id}       ->  full Session (history, tools, ...)  sliding TTL on activity
```

On each request:

1. Read bearer token from the cookie (the token IS the cookie value -- no signing key)
2. Look up `tok:{token}` to dereference to `user_id` (one store read)
3. Look up `uid:{user_id}` to get the full session with history (second store read)

On re-authentication (`/callback`):

- New `tok:{new_token}` entry is written pointing to `user_id`
- Existing `uid:{user_id}` entry is updated with the new token; history is preserved
- Cached tool list is invalidated (`tools = None`) so it re-fetches with the new token
- Old `tok:{old_token}` entry expires naturally via TTL sweep

On logout:

- Both `tok:{token}` and `uid:{user_id}` entries are deleted

**Why not a separate signing key?**
The cookie value is an opaque bearer token issued by Credenza. Any forged cookie
value would be rejected by the store lookup (no `tok:` entry). `HttpOnly + Secure +
SameSite=Lax` prevent theft and CSRF. No signing key is needed -- and eliminating it
removes the `chatbot_session_secret` Docker secret entirely.

**Token TTL vs session TTL:**
The `tok:` entry expires after `session_ttl` seconds from login, matching the
Credenza token lifetime (operators should align `DERIVA_CHATBOT_SESSION_TTL` with the
token TTL configured on the client registration). The `uid:` entry uses sliding TTL --
it is refreshed on every active request -- so conversation history persists as long as
the user is active, surviving multiple re-authentication cycles.

---

### Phase 8 -- Token Efficiency, Hardening & UI Polish

**Status:** Complete.

#### Token efficiency

- **Tool schema slimming** (`mcp_client._slim_input_schema`): strips `description` and
  `title` from every property definition in `input_schema.properties` before caching the
  tool list. Top-level schema fields are preserved. Reduces fixed tool-list token cost by
  ~30-50% without affecting Claude's ability to call tools.

- **Prompt caching**: system prompt and tool list are passed with
  `cache_control: {"type": "ephemeral"}` on every API call. Anthropic caches the
  tokenized blocks server-side for 5 minutes; cache hits are charged at 0.1x normal input
  token cost and count minimally toward TPM limits. Effectively eliminates the fixed
  ~25k-token tool-list cost on all turns after the first within a session.

- **Three-tier tool result truncation**:
    - `_TOOL_RESULT_PREVIEW = 1000` chars -- shown in the UI tool call block
    - `_TOOL_RESULT_TO_CLAUDE = 10000` chars -- fed back to Claude in the current turn
      (raised from 6000 after testing showed 10 study rows were being truncated)
    - `_HISTORY_TOOL_RESULT_MAX = 3000` chars -- stored in session history for subsequent turns

- **Model default**: Haiku (`claude-haiku-4-5-latest`). Haiku has a 200k token/min rate
  limit (vs ~30k for Sonnet at the free tier) and is sufficient for DERIVA data assistant
  workloads after prompt engineering hardening (guide injection, mandatory rules in system
  prompt). Override with `DERIVA_CHATBOT_CLAUDE_MODEL`.

- **Retry with exponential backoff**: `run_chat_turn` retries on HTTP 429 (rate limit)
  and 529 (overloaded) with delays of 5s, 10s, 20s (up to `_MAX_API_RETRIES = 3`).
  Retry only fires if no text has been yielded yet in that loop iteration -- once text is
  in-flight to the SSE client it cannot be retracted, so errors mid-stream propagate
  immediately.

#### Hardening

- **History corruption fix** (`trim_history`): the slice-and-advance loop now skips any
  tail segment that begins on an orphaned `tool_result` user message (whose corresponding
  `tool_use` was trimmed away). Without this fix, history trimming at turn boundaries
  caused persistent API 400 errors on all subsequent turns.

- **Poll delay enforcement**: when the same tool is called again in the next tool-calling
  loop iteration (background task polling pattern), `asyncio.sleep(_POLL_DELAY_SECONDS)`
  (5s) is inserted before executing the tool. Claude says it will wait but cannot actually
  sleep -- the delay is enforced in the loop regardless of what Claude says.

- **Display rules in system prompt**: the "always show all columns including null-value
  columns; always include RID; omit RCT/RMT/RCB/RMB unless requested" display rules are
  injected into the system prompt (not just the tool descriptions). Additionally, the
  system prompt now includes five MANDATORY RULES at the top (schema context usage,
  multi-table join guidance, zero-result acceptance, display rules, and operational rules)
  that are reinforced by the MCP guide prompts injected from `tools/prompts.py` in
  `deriva-mcp-core`. This layered approach (system prompt rules + guide prompts) works
  reliably with both Haiku and Sonnet.

- **RFC 7009 token revocation on logout**: `logout()` POSTs to `{credenza_url}/revoke`
  with `token` and `client_id` after clearing server-side session state. Best-effort --
  a revocation failure is logged but does not block the local logout.

- **Python logging** (`server._init_logging`): stderr `StreamHandler` + optional
  `SysLogHandler` to `/dev/log` (LOCAL1 facility) following the deriva-mcp-core pattern.
  `httpx` and `httpcore` suppressed to WARNING. Called in `main()` with debug flag.

#### UI polish

- **Left-aligned chat layout**: both user and assistant messages are left-aligned and
  interleaved, replacing the right-aligned user bubble. User messages use a steel-blue
  tint (`#b8d0ed`) to distinguish them from the gray assistant bubbles (`#e8ecf0`).

- **Collapsible tool container**: all tool call blocks for a turn are wrapped in a parent
  `<details>` that starts collapsed, showing only "N tool calls" in the summary. Individual
  tool calls remain independently expandable inside. Container is hidden when no tools ran.

- **Smart paragraph separator**: when Claude emits text, calls a tool, then emits more
  text, a `"\n\n"` paragraph break is injected between the segments if the preceding text
  ends with sentence-ending punctuation (`.`, `!`, `?`, `:`); otherwise a single space is
  injected to handle mid-sentence splits gracefully.

- **Incremental Markdown rendering**: text chunks are rendered through `marked.parse()`
  incrementally during streaming, not as raw text. Final render on stream end.

- **Prose/structure visual separation**: CSS adjacent sibling selector adds a rule line
  between paragraph text and the first structural markdown element (list, table, heading)
  that follows it within an assistant message.

---

### Phase 9 -- Multi-Conversation History (Design Note)

**Status:** Not started. Design only.

---

#### Motivation

The current model supports one active conversation per user. A history sidebar --
as used by Claude.ai, ChatGPT, etc. -- lets users maintain multiple named
conversations and switch between them. For a DERIVA data assistant this is
genuinely useful: a user might have one conversation exploring a subject table,
another investigating the image pipeline, and want to return to either without
losing context.

A related improvement is token-count-aware history trimming: the current
turn-count trim (`max_history_turns`) is a blunt instrument. With heavy tool use
each turn can contain large result payloads, so 20 turns can easily exceed 100K
tokens -- expensive per API call and slow. Trimming by token budget rather than
turn count is more accurate and naturally falls out of the multi-conversation
redesign.

---

#### Context size

Every Claude API call sends the full history. With `max_history_turns=20`:

- User message: up to `max_message_length` chars (~2,500 tokens at default)
- Assistant response: up to `_MAX_TOKENS` (8,192 tokens)
- Tool results: unbounded -- a `get_entities` dump can be 10-50 KB per call,
  multiple per turn

A 20-turn tool-heavy conversation can send 100K+ tokens per request. The
Anthropic SDK provides `client.messages.count_tokens()` for an exact count.
A practical approach: after each turn, count the tokens in the accumulated
history and trim oldest pairs until the count is under a configurable budget
(e.g., `DERIVA_CHATBOT_MAX_CONTEXT_TOKENS=80000`). This replaces or supplements
`max_history_turns`.

Storage size is not the concern -- 200 KB of JSON per conversation is nothing
for any persistent backend. The constraint is the per-request token cost.

---

#### Data model

The key-value store pattern extends naturally without changing the `SessionStore`
protocol. New key namespaces alongside the existing `tok:` and `uid:` entries:

```
tok:{bearer_token}       ->  {user_id}               (existing, TTL = session_ttl)
uid:{user_id}            ->  {current_conv_id, ...}  (extended -- auth state only)
conv:{conv_id}           ->  Conversation JSON        (history, tools, schema_primed)
convs:{user_id}          ->  ConversationIndex JSON   (ordered list of metadata)
```

`uid:{user_id}` loses the history fields -- it becomes a lightweight auth record
that tracks the current conversation ID and last_active. History moves entirely
into `conv:{conv_id}`.

`convs:{user_id}` stores a JSON list of conversation metadata:

```json
[
  {
    "conv_id": "c-abc123",
    "title": "Exploring the Subject table",
    "created_at": 1743000000.0,
    "last_active": 1743003600.0,
    "message_count": 14
  },
  ...
]
```

This is written on every conversation update. For most users the list will be
short (10-50 entries) so a single key read is fine. At large scale (1000s of
conversations) a database-backed secondary index would be more appropriate, but
that is outside the scope of this phase.

`conv:{conv_id}` stores:

```json
{
  "conv_id": "c-abc123",
  "user_id": "alice@example.org",
  "title": "Exploring the Subject table",
  "created_at": 1743000000.0,
  "last_active": 1743003600.0,
  "history": [
    ...
  ],
  "tools": null,
  "schema_primed": false
}
```

---

#### API changes

```
GET  /conversations                    list conversations for current user
POST /conversations                    create new conversation -> {conv_id, title}
DELETE /conversations/{conv_id}        delete a conversation

POST /chat                             body gains optional conv_id field;
                                       omit to use current conversation,
                                       provide to switch to a specific one
```

`GET /conversations` returns the `convs:{user_id}` index, sorted by
`last_active` descending.

`POST /conversations` generates a `conv_id` (e.g., `c-` + `secrets.token_urlsafe(12)`),
creates a new `conv:` entry with empty history, appends metadata to `convs:`,
and sets it as the current conversation on `uid:`.

`POST /chat` with a `conv_id` loads `conv:{conv_id}` instead of the current
conversation. The server verifies `conv.user_id == session.user_id` before
serving it.

---

#### Title generation

Two options, in order of cost:

1. **Truncate first user message** -- take the first 60 characters of the
   opening message. Zero latency, zero cost, good enough for most cases.

2. **Async Claude summarization** -- after the first turn completes, fire a
   background task calling Claude with a short prompt ("Summarize this question
   in 6 words or fewer: ..."). Store the result back into the conversation
   metadata. The sidebar updates on next page render.

Option 1 is sufficient for v1 of this phase. Option 2 can be layered on using
the existing `TaskManager` pattern from `deriva-mcp-core` if desired.

---

#### UI changes

The current single-column layout gains a left sidebar:

```
+--sidebar--+------------------main-----------------+
| [+ New]   |  header                               |
|           +---------------------------------------+
| Conv A    |  message thread                       |
| Conv B    |                                       |
| Conv C    |                                       |
|           +---------------------------------------+
|           |  input area                           |
+-----------+---------------------------------------+
```

The sidebar is hidden on narrow viewports (mobile) via a hamburger toggle.
Clicking a conversation entry issues `POST /chat` with the target `conv_id`
(or a `GET /conversations/{conv_id}` to load history for display, then
switches the active context).

On desktop the sidebar can collapse to icon-only to maximize the chat area.

---

#### Migration from Phase 7

Existing `uid:{user_id}` entries contain a full Session with history.
On first request after upgrade, if the entry has a non-empty `history` field
and no `current_conv_id`, the server migrates it:

1. Generate a `conv_id` for the legacy conversation.
2. Write `conv:{conv_id}` with the existing history.
3. Write `convs:{user_id}` with a single metadata entry, title = "Previous
   conversation", `created_at` = session `created_at`.
4. Rewrite `uid:{user_id}` as a lightweight auth record with `current_conv_id`.

This migration runs in-place in `require_session` and is transparent to the
user: they land on their existing conversation as if nothing changed, with the
sidebar showing one entry.

---

#### Deployment note

`convs:{user_id}` is written on every conversation update. For the memory
backend this is in-process and free. For Redis/Valkey this is one extra `SET`
per chat turn. For PostgreSQL/SQLite it may be worth storing conversation
metadata in a separate `chatbot_conversations` table rather than as a JSON blob
in the key-value store, to allow efficient listing and deletion without
deserializing all history. This is a backend-specific optimization and does not
affect the API or UI.

---

### Post-Phase-8 Hardening

**Anonymous / zero-auth mode [DONE -- 2026-04-02]**

Added support for deployments where Credenza is not configured. When
`DERIVA_CHATBOT_CREDENZA_URL` is not set, the UI operates in anonymous mode:
no login is required and sessions are browser-scoped using a random cookie.

#### Key design decisions

- **Auto-detection, no explicit mode flag.** `Settings.auth_enabled` is a
  computed property: `bool(self.credenza_url)`. If `credenza_url` is set,
  full OAuth is required. If not, anonymous mode is active. This mirrors
  the `DERIVA_MCP_ALLOW_ANONYMOUS` pattern in `deriva-mcp-core`.

- **Per-browser anonymous sessions.** Each browser gets an opaque random ID
  (`deriva_chatbot_anon` cookie). The session key is `uid:anonymous/{anon_id}`.
  Sessions persist for `session_ttl` seconds (same as authenticated sessions).
  A new session is created on first request; subsequent requests read the
  existing one.

- **No bearer token.** `Session.bearer_token` is `str | None = None`.
  `mcp_client.py` omits the `Authorization` header entirely when the token
  is None.

- **Validated config paths.**
    - `auth_enabled=True`: all Credenza fields required at startup (unchanged).
    - `auth_enabled=False`: only `mcp_url` and `anthropic_api_key` required;
      Credenza fields not validated.

#### Cookie delivery -- middleware workaround

FastAPI does not merge cookies set on an injected `Response` dependency into
the actual HTTP response when the endpoint returns `JSONResponse` directly.
The workaround: `_get_or_create_anonymous_session()` stores the new cookie
info in `request.state.new_anon_id = (anon_id, max_age)`, and
`_anon_cookie_middleware` (registered as `@app.middleware("http")`) reads
it and attaches the `Set-Cookie` header to the outgoing response.

---

## Out of Scope

- Multi-user conversation sharing or persistence across browser sessions (each session
  is private to the authenticated user)
- Direct Hatrac file upload/download via the UI (tool calls can reference Hatrac paths
  but the UI does not handle binary transfers)
- Customizable system prompts per-user (operator-set only via config)
- WebSocket transport (SSE over HTTP is sufficient and simpler to proxy)
- Any modification to `deriva-mcp-core` -- the MCP server is consumed as-is