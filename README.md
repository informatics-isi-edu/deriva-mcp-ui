# deriva-mcp-ui

[![CI Status](https://github.com/informatics-isi-edu/deriva-mcp-ui/actions/workflows/ci.yaml/badge.svg)](https://github.com/informatics-isi-edu/deriva-mcp-ui/actions/workflows/ci.yaml)
[![Coverage Status](https://coveralls.io/repos/github/informatics-isi-edu/deriva-mcp-ui/badge.svg?branch=main)](https://coveralls.io/github/informatics-isi-edu/deriva-mcp-ui?branch=main)

Browser-based chatbot interface for DERIVA. Wraps Claude and the
[deriva-mcp-core](https://github.com/informatics-isi-edu/deriva-mcp-core) MCP server
behind a web frontend, giving end users a natural-language interface to query and manage
DERIVA catalogs without needing a desktop MCP client.

## Features

- Standard web login via Credenza (OAuth 2.0 + PKCE -- no token pasting)
- Streaming responses via Server-Sent Events (Claude's replies appear word-by-word)
- Full tool-calling loop: Claude invokes DERIVA tools transparently; tool calls are
  shown in collapsible blocks in the UI
- Two operating modes:
    - **Default-catalog** -- anchored to a specific catalog; schema context injected
      automatically on the first turn
    - **General-purpose** -- user or Claude specifies hostname and catalog ID
- Conversation history preserved server-side across the browser session and across
  Credenza token re-authentication
- Multiple session storage backends: memory (default), Redis/Valkey, PostgreSQL, SQLite

## Requirements

- Python 3.11+
- A running [deriva-mcp-core](https://github.com/informatics-isi-edu/deriva-mcp-core)
  instance reachable over HTTP
- A [Credenza](https://github.com/informatics-isi-edu/credenza) instance for
  authentication
- An Anthropic API key

## Installation

```bash
pip install deriva-mcp-ui
# or with uv:
uv add deriva-mcp-ui
```

With optional backends:

```bash
pip install "deriva-mcp-ui[redis]"      # Redis / Valkey
pip install "deriva-mcp-ui[sqlite]"     # SQLite
pip install "deriva-mcp-ui[postgresql]" # PostgreSQL
```

## Configuration

All configuration is via environment variables with the `DERIVA_CHATBOT_` prefix.

### Required

| Variable                      | Description                                                    |
|-------------------------------|----------------------------------------------------------------|
| `DERIVA_CHATBOT_MCP_URL`      | Base URL of the deriva-mcp-core server                         |
| `DERIVA_CHATBOT_CREDENZA_URL` | Base URL of the Credenza instance                              |
| `DERIVA_CHATBOT_CLIENT_ID`    | OAuth client ID registered in Credenza                         |
| `DERIVA_CHATBOT_MCP_RESOURCE` | Resource URI for the MCP server (must match MCP server config) |
| `DERIVA_CHATBOT_PUBLIC_URL`   | Public HTTPS URL of this service (used as OAuth redirect base) |
| `ANTHROPIC_API_KEY`           | Anthropic API key                                              |

### Default-catalog mode

Set both variables to anchor the chatbot to a specific catalog:

| Variable                               | Description                                    |
|----------------------------------------|------------------------------------------------|
| `DERIVA_CHATBOT_DEFAULT_HOSTNAME`      | DERIVA server hostname                         |
| `DERIVA_CHATBOT_DEFAULT_CATALOG_ID`    | Catalog ID or alias                            |
| `DERIVA_CHATBOT_DEFAULT_CATALOG_LABEL` | Display name shown in the UI header (optional) |

### Tuning

| Variable                             | Default            | Description                                                          |
|--------------------------------------|--------------------|----------------------------------------------------------------------|
| `DERIVA_CHATBOT_CLAUDE_MODEL`        | `claude-haiku-4-5` | Claude model ID                                                      |
| `DERIVA_CHATBOT_MAX_HISTORY_TURNS`   | `10`               | Conversation turns retained per session                              |
| `DERIVA_CHATBOT_MAX_MESSAGE_LENGTH`  | `10000`            | Maximum user message length in characters                            |
| `DERIVA_CHATBOT_SESSION_TTL`         | `28800`            | Server-side session TTL in seconds (default 8h)                      |
| `DERIVA_CHATBOT_STORAGE_BACKEND`     | `memory`           | Session backend: `memory`, `redis`, `valkey`, `postgresql`, `sqlite` |
| `DERIVA_CHATBOT_STORAGE_BACKEND_URL` | --                 | Connection URL for the selected backend                              |
| `DERIVA_CHATBOT_DEBUG`               | `false`            | Enable debug logging                                                 |

Storage URL examples:

```
redis://localhost:6379/0
postgresql://user:pass@host/dbname
sqlite:///path/to/sessions.db
```

## Running

```bash
deriva-mcp-ui
```

The server listens on `0.0.0.0:8001`. Set `DERIVA_CHATBOT_DEBUG=true` for debug logging.

## Docker

```dockerfile
FROM ghcr.io/informatics-isi-edu/deriva-mcp-ui:latest
```

Or build locally:

```bash
docker build -t deriva-mcp-ui .
```

Example `docker-compose.yml` snippet:

```yaml
deriva-mcp-ui:
  image: deriva-mcp-ui:latest
  environment:
    DERIVA_CHATBOT_MCP_URL: http://deriva-mcp-core:8000
    DERIVA_CHATBOT_CREDENZA_URL: https://auth.example.org
    DERIVA_CHATBOT_CLIENT_ID: deriva-mcp-ui
    DERIVA_CHATBOT_MCP_RESOURCE: https://mcp.example.org
    DERIVA_CHATBOT_PUBLIC_URL: https://example.org/chatbot
    ANTHROPIC_API_KEY: sk-ant-...
    DERIVA_CHATBOT_DEFAULT_HOSTNAME: data.example.org
    DERIVA_CHATBOT_DEFAULT_CATALOG_ID: "1"
    DERIVA_CHATBOT_DEFAULT_CATALOG_LABEL: "Example Catalog"
    DERIVA_CHATBOT_STORAGE_BACKEND: redis
    DERIVA_CHATBOT_STORAGE_BACKEND_URL: redis://redis:6379/0
```

## Deployment

### Traefik (deriva-docker)

Add a `deriva-mcp-ui` service with these labels:

```yaml
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.chatbot.rule=PathPrefix(`/chatbot`)"
  - "traefik.http.middlewares.chatbot-strip.stripprefix.prefixes=/chatbot"
  - "traefik.http.routers.chatbot.middlewares=chatbot-strip"
  - "traefik.http.services.chatbot.loadbalancer.server.port=8001"
```

Access the UI at `https://your-host/chatbot/`. Note the trailing slash -- the browser
must include it so relative asset URLs resolve correctly.

### Apache (VM deployment)

```apache
ProxyPass /chatbot/ http://127.0.0.1:8001/
ProxyPassReverse /chatbot/ http://127.0.0.1:8001/
```

### Credenza client registration

Add an entry to `config/client_registry.json`:

```json
{
  "deriva-chatbot": {
    "desc": "DERIVA Chatbot (public, PKCE-only)",
    "enabled": true,
    "public": true,
    "allowed_grant_types": [
      "authorization_code"
    ],
    "allowed_redirect_uris": [
      "https://your-host.example.org/chatbot/callback",
      "http://localhost:8001/callback"
    ],
    "allowed_resources": [
      "urn:deriva:rest:service:all",
      "https://your-host.example.org/mcp"
    ],
    "allowed_scopes": [
      "openid"
    ],
    "require_consent": true,
    "consent_display_name": "DERIVA Chatbot Server",
    "consent_labels": {
      "https://your-host.example.org/mcp": "Connect to the DERIVA MCP Server at this URL"
    },
    "max_session_ttl_seconds": 28800
  }
}
```

Set `DERIVA_CHATBOT_MCP_RESOURCE` to the resource URI listed in `allowed_resources`
(e.g. `https://your-host.example.org/mcp`). Align `DERIVA_CHATBOT_SESSION_TTL` with
`max_session_ttl_seconds`. Conversation history is preserved across token expiry and
re-authentication -- users return to their existing conversation after logging in again.

## Architecture

```
Browser (HTML + JS)
  |
  | HTTPS (session cookie)
  v
deriva-mcp-ui  (FastAPI, port 8001)
  |                    |
  | MCP over HTTP      | HTTPS
  | (bearer token)     | (Anthropic API)
  v                    v
deriva-mcp-core     Claude
  |
  | HTTPS
  v
DERIVA (ERMrest, Hatrac)
```

The UI service is a stateless MCP client: each chat turn opens a fresh HTTP connection
to the MCP server, runs the Claude tool-calling loop, streams text back via SSE, then
closes. No persistent MCP session is maintained.

## Development

```bash
git clone https://github.com/informatics-isi-edu/deriva-mcp-ui
cd deriva-mcp-ui
uv sync --extra dev
uv run pytest
```

Lint:

```bash
uv run ruff check src tests
```

## Health Endpoint

The HTTP server exposes a health endpoint at `GET /health` that returns `{"status": "ok"}`
with no authentication required. Suitable for Docker health probes and load balancer checks.

```bash
curl http://localhost:8001/health
# {"status":"ok"}
```

## Development Status

`deriva-mcp-ui` is alpha-quality software. The API and configuration surface are
still evolving and breaking changes may occur between releases without advance notice.
It has been validated end-to-end against live DERIVA deployments, but has not yet seen
broad production use. Use in production environments is at your own risk. Bug reports
and contributions are welcome via the
[issue tracker](https://github.com/informatics-isi-edu/deriva-mcp-ui/issues).

## License

Apache 2.0. See [LICENSE](LICENSE).
