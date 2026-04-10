"""Tests for auth.py: PKCE helpers, session key pattern, route logic."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from deriva_mcp_ui.auth import (
    _code_challenge,
    _generate_code_verifier,
    _token_key,
    history_key,
    user_session_key,
)
from deriva_mcp_ui.config import Settings
from deriva_mcp_ui.server import create_app
from deriva_mcp_ui.storage import Session
from deriva_mcp_ui.storage.memory import MemorySessionStore


def _test_settings(**kwargs) -> Settings:
    base = dict(
        mcp_url="http://mcp:8000",
        credenza_url="http://credenza",
        client_id="test-client",
        mcp_resource="https://mcp.example.org",
        public_url="https://chatbot.example.org",
        llm_api_key="sk-test",
    )
    base.update(kwargs)
    return Settings(**base)


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def test_code_verifier_is_base64url():
    v = _generate_code_verifier()
    assert len(v) > 0
    assert "=" not in v
    assert "+" not in v
    assert "/" not in v


def test_code_challenge_s256():
    import hashlib
    from base64 import urlsafe_b64encode

    verifier = "test-verifier-abc123"
    expected = urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert _code_challenge(verifier) == expected


# ---------------------------------------------------------------------------
# Session key helpers
# ---------------------------------------------------------------------------


def test_token_key_prefix():
    assert _token_key("abc123") == "tok:abc123"


def test_user_session_key_prefix():
    assert user_session_key("alice@example.org") == "uid:alice@example.org"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_and_store():
    settings = _test_settings()
    app = create_app(settings)
    store = MemorySessionStore(ttl=settings.session_ttl)
    app.state.store = store
    return app, store


def _mock_http(token_resp_json: dict, session_resp_json: dict):
    """Context manager that patches httpx.AsyncClient for token exchange + GET /session."""
    mock_token_resp = MagicMock()
    mock_token_resp.status_code = 200
    mock_token_resp.json.return_value = token_resp_json

    mock_session_resp = MagicMock()
    mock_session_resp.status_code = 200
    mock_session_resp.json.return_value = session_resp_json

    async def mock_post(url, **kwargs):
        if "/token" in url:
            return mock_token_resp
        raise AssertionError(f"Unexpected POST URL: {url}")

    async def mock_get(url, **kwargs):
        if "/session" in url:
            return mock_session_resp
        raise AssertionError(f"Unexpected GET URL: {url}")

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post
    mock_client.get = mock_get
    return patch("deriva_mcp_ui.auth.httpx.AsyncClient", return_value=mock_client)


# ---------------------------------------------------------------------------
# /login route
# ---------------------------------------------------------------------------


def test_login_redirects_to_credenza(app_and_store):
    app, _ = app_and_store
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/login")
    assert resp.status_code == 302
    loc = resp.headers["location"]
    assert "http://credenza/authorize" in loc
    assert "code_challenge=" in loc
    assert "state=" in loc
    # Both resources must be present: MCP resource + DERIVA service resource
    assert loc.count("resource=") == 2
    assert "urn%3Aderiva%3Arest%3Aservice%3Aall" in loc
    assert "deriva_chatbot_pkce" in resp.cookies


# ---------------------------------------------------------------------------
# /callback route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_state_mismatch(app_and_store):
    app, _ = app_and_store
    client = TestClient(app, follow_redirects=False)
    pkce = json.dumps({"verifier": "v", "state": "aaa"})
    resp = client.get("/callback?code=abc&state=bbb", cookies={"deriva_chatbot_pkce": pkce})
    assert resp.status_code == 400
    assert "State mismatch" in resp.text


@pytest.mark.asyncio
async def test_callback_missing_pkce_cookie(app_and_store):
    app, _ = app_and_store
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/callback?code=abc&state=xyz")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_callback_oauth_error(app_and_store):
    app, _ = app_and_store
    client = TestClient(app, follow_redirects=False)
    pkce = json.dumps({"verifier": "v", "state": "s"})
    resp = client.get("/callback?error=access_denied&state=s", cookies={"deriva_chatbot_pkce": pkce})
    assert resp.status_code == 400
    assert "access_denied" in resp.text


@pytest.mark.asyncio
async def test_callback_successful_flow(app_and_store):
    app, store = app_and_store
    state = "test-state-value"
    pkce = json.dumps({"verifier": "test-verifier", "state": state})

    with _mock_http(
        {"access_token": "bearer-tok-123"},
        {"client": {"id": "user@example.org", "display_name": "Test User"}, "active": True},
    ):
        client = TestClient(app, follow_redirects=False)
        resp = client.get(
            f"/callback?code=auth-code&state={state}",
            cookies={"deriva_chatbot_pkce": pkce},
        )

    assert resp.status_code == 302
    assert resp.headers["location"] == "https://chatbot.example.org/"
    # Cookie value is the bearer token directly
    assert resp.cookies["deriva_chatbot_session"] == "bearer-tok-123"
    # Token index entry stored
    tok_entry = await store.get(_token_key("bearer-tok-123"))
    assert tok_entry is not None
    assert tok_entry.user_id == "user@example.org"
    # Full session stored by user_id
    session = await store.get(user_session_key("user@example.org"))
    assert session is not None
    assert session.user_id == "user@example.org"
    assert session.bearer_token == "bearer-tok-123"
    assert session.credenza_session.get("client", {}).get("display_name") == "Test User"


@pytest.mark.asyncio
async def test_callback_preserves_history_on_reauth(app_and_store):
    """Re-authentication updates the token but preserves existing conversation history."""
    app, store = app_and_store

    # Seed an existing session as if the user previously logged in and had a conversation
    old_bearer = "old-bearer-token"
    history = [
        {"role": "user", "content": "What tables exist?"},
        {"role": "assistant", "content": "The catalog has tables: ..."},
    ]
    existing = Session(
        user_id="alice@example.org",
        bearer_token=old_bearer,
        history=history,
        created_at=time.time(),
        last_active=time.time(),
    )
    await store.set(user_session_key("alice@example.org"), existing)
    await store.set(
        _token_key(old_bearer),
        Session(
            user_id="alice@example.org",
            bearer_token=old_bearer,
            created_at=time.time(),
            last_active=time.time(),
        ),
    )

    # User re-authenticates with a new token
    state = "reauth-state"
    pkce = json.dumps({"verifier": "reauth-verifier", "state": state})

    with _mock_http(
        {"access_token": "new-bearer-token"},
        {"client": {"id": "alice@example.org", "display_name": "Alice"}, "active": True},
    ):
        client = TestClient(app, follow_redirects=False)
        resp = client.get(
            f"/callback?code=new-auth-code&state={state}",
            cookies={"deriva_chatbot_pkce": pkce},
        )

    assert resp.status_code == 302
    assert resp.cookies["deriva_chatbot_session"] == "new-bearer-token"

    # History is preserved in the user session
    session = await store.get(user_session_key("alice@example.org"))
    assert session is not None
    assert session.bearer_token == "new-bearer-token"
    assert session.history == history
    # tools cache invalidated so it re-fetches with the new token
    assert session.tools is None
    # credenza_session updated with new identity data
    assert session.credenza_session.get("client", {}).get("id") == "alice@example.org"


# ---------------------------------------------------------------------------
# /logout route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logout_clears_session(app_and_store):
    app, store = app_and_store

    bearer = "tok-logout-test"
    now = time.time()
    session = Session(user_id="alice", bearer_token=bearer, created_at=now, last_active=now)
    await store.set(user_session_key("alice"), session)
    await store.set(
        _token_key(bearer),
        Session(user_id="alice", bearer_token=bearer, created_at=now, last_active=now),
    )

    from unittest.mock import AsyncMock, MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 302
    mock_response.headers = {"location": "https://idp.example.com/logout?post_logout_redirect_uri=https://app/"}

    with patch("deriva_mcp_ui.auth.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = TestClient(app, follow_redirects=False)
        resp = client.get("/logout", cookies={"deriva_chatbot_session": bearer})

    assert resp.status_code == 302
    # Credenza logout called with bearer token in Authorization header
    call_kwargs = mock_http.get.call_args
    assert call_kwargs.kwargs["headers"]["Authorization"] == f"Bearer {bearer}"
    assert call_kwargs.kwargs["follow_redirects"] is False
    # Browser redirected to IDP logout URL from Credenza's Location header
    assert resp.headers["location"] == "https://idp.example.com/logout?post_logout_redirect_uri=https://app/"

    assert await store.get(user_session_key("alice")) is None
    assert await store.get(_token_key(bearer)) is None


@pytest.mark.asyncio
async def test_logout_handles_credenza_legacy_mode(app_and_store):
    """Legacy Credenza returns 200 + {logout_url: ...}; browser is redirected there."""
    app, store = app_and_store

    bearer = "tok-legacy-logout"
    now = time.time()
    await store.set(user_session_key("legacy-user"), Session(user_id="legacy-user", bearer_token=bearer, created_at=now, last_active=now))
    await store.set(_token_key(bearer), Session(user_id="legacy-user", bearer_token=bearer, created_at=now, last_active=now))

    from unittest.mock import AsyncMock, MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json = MagicMock(return_value={"logout_url": "https://idp.example.com/logout"})

    with patch("deriva_mcp_ui.auth.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = TestClient(app, follow_redirects=False)
        resp = client.get("/logout", cookies={"deriva_chatbot_session": bearer})

    assert resp.status_code == 302
    assert resp.headers["location"] == "https://idp.example.com/logout"
    assert await store.get(user_session_key("legacy-user")) is None


# ---------------------------------------------------------------------------
# require_session dependency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_info_no_cookie(app_and_store):
    app, _ = app_and_store
    client = TestClient(app)
    resp = client.get("/session-info")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_session_info_with_valid_session(app_and_store):
    app, store = app_and_store

    bearer = "tok-info-test"
    now = time.time()
    session = Session(user_id="bob", bearer_token=bearer, created_at=now, last_active=now)
    await store.set(user_session_key("bob"), session)
    await store.set(
        _token_key(bearer),
        Session(user_id="bob", bearer_token=bearer, created_at=now, last_active=now),
    )

    client = TestClient(app)
    resp = client.get("/session-info", cookies={"deriva_chatbot_session": bearer})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "bob"
    assert data["catalog_mode"] == "general"


@pytest.mark.asyncio
async def test_session_info_stale_token(app_and_store):
    """Cookie present but token index expired -- 401 expected."""
    app, store = app_and_store

    # User session exists but token index is gone (token expired/swept)
    bearer = "expired-tok"
    now = time.time()
    session = Session(user_id="carol", bearer_token=bearer, created_at=now, last_active=now)
    await store.set(user_session_key("carol"), session)
    # tok: entry intentionally not seeded

    client = TestClient(app)
    resp = client.get("/session-info", cookies={"deriva_chatbot_session": bearer})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_session_info_tok_entry_but_no_uid_session(app_and_store):
    """tok: entry exists but uid: entry is absent -- 401 expected (line 133)."""
    app, store = app_and_store

    bearer = "tok-orphan"
    now = time.time()
    # Write tok: entry pointing to a user_id that has no uid: entry
    await store.set(
        _token_key(bearer),
        Session(user_id="ghost", bearer_token=bearer, created_at=now, last_active=now),
    )
    # uid:ghost intentionally absent

    client = TestClient(app)
    resp = client.get("/session-info", cookies={"deriva_chatbot_session": bearer})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_callback_invalid_pkce_json(app_and_store):
    """Malformed PKCE cookie JSON returns 400 (lines 202-203)."""
    app, _ = app_and_store
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/callback?code=abc&state=s", cookies={"deriva_chatbot_pkce": "not-json"})
    assert resp.status_code == 400
    assert "Invalid PKCE cookie" in resp.text


@pytest.mark.asyncio
async def test_callback_token_exchange_failure(app_and_store):
    """Non-200 token exchange returns 502 (lines 225-226)."""
    app, _ = app_and_store
    state = "fail-state"
    pkce = json.dumps({"verifier": "v", "state": state})
    with _mock_http({"status": 503}, {}):
        # Patch the token response status directly after the mock is set up
        pass

    # Use _mock_http but override the token resp status to 503
    mock_token_resp = MagicMock()
    mock_token_resp.status_code = 503
    mock_session_resp = MagicMock()
    mock_session_resp.status_code = 200
    mock_session_resp.json.return_value = {}

    async def mock_post(url, **kwargs):
        return mock_token_resp

    async def mock_get(url, **kwargs):
        return mock_session_resp

    from unittest.mock import AsyncMock, patch

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post
    mock_client.get = mock_get

    with patch("deriva_mcp_ui.auth.httpx.AsyncClient", return_value=mock_client):
        client = TestClient(app, follow_redirects=False)
        resp = client.get(
            f"/callback?code=abc&state={state}",
            cookies={"deriva_chatbot_pkce": pkce},
        )

    assert resp.status_code == 502
    assert "Token exchange failed" in resp.text


@pytest.mark.asyncio
async def test_callback_no_access_token_in_response(app_and_store):
    """Token exchange succeeds but response has no access_token -- 502 (line 231)."""
    app, _ = app_and_store
    state = "notoken-state"
    pkce = json.dumps({"verifier": "v", "state": state})

    mock_token_resp = MagicMock()
    mock_token_resp.status_code = 200
    mock_token_resp.json.return_value = {}  # no access_token or token key

    async def mock_post(url, **kwargs):
        return mock_token_resp

    from unittest.mock import AsyncMock, patch

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post

    with patch("deriva_mcp_ui.auth.httpx.AsyncClient", return_value=mock_client):
        client = TestClient(app, follow_redirects=False)
        resp = client.get(
            f"/callback?code=abc&state={state}",
            cookies={"deriva_chatbot_pkce": pkce},
        )

    assert resp.status_code == 502
    assert "access_token" in resp.text


# ---------------------------------------------------------------------------
# Anonymous mode
# ---------------------------------------------------------------------------


@pytest.fixture()
def anon_app_and_store():
    # No credenza_url -> anonymous mode (auth_enabled=False)
    settings = Settings(
        mcp_url="http://mcp:8000",
        llm_api_key="sk-test",
    )
    app = create_app(settings)
    store = MemorySessionStore(ttl=settings.session_ttl)
    app.state.store = store
    return app, store


def test_login_anonymous_mode_redirects_to_root(anon_app_and_store):
    """/login in anonymous mode redirects to / without starting OAuth."""
    app, _ = anon_app_and_store
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/login")
    assert resp.status_code == 302
    assert resp.headers["location"] == "/"
    # No PKCE cookie set
    assert "deriva_chatbot_pkce" not in resp.cookies


@pytest.mark.asyncio
async def test_anonymous_session_created_on_first_request(anon_app_and_store):
    """First request without a cookie creates a new anonymous session."""
    app, store = anon_app_and_store
    client = TestClient(app, follow_redirects=False)
    resp = client.get("/session-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["display_name"] == "Anonymous"
    assert data["user_id"].startswith("anonymous/")
    # Cookie was set
    assert "deriva_chatbot_anon" in resp.cookies


@pytest.mark.asyncio
async def test_anonymous_session_persists_across_requests(anon_app_and_store):
    """Second request with the anon cookie reuses the existing session."""
    app, store = anon_app_and_store
    client = TestClient(app, follow_redirects=False)

    # First request -- creates session
    resp1 = client.get("/session-info")
    assert resp1.status_code == 200
    anon_id = resp1.cookies["deriva_chatbot_anon"]
    user_id_1 = resp1.json()["user_id"]

    # Second request with the cookie -- same session
    resp2 = client.get("/session-info", cookies={"deriva_chatbot_anon": anon_id})
    assert resp2.status_code == 200
    assert resp2.json()["user_id"] == user_id_1


@pytest.mark.asyncio
async def test_anonymous_logout_clears_cookie(anon_app_and_store):
    """Logout in anonymous mode deletes the anon session and clears the cookie."""
    app, store = anon_app_and_store
    client = TestClient(app, follow_redirects=False)

    # Create a session
    resp1 = client.get("/session-info")
    anon_id = resp1.cookies["deriva_chatbot_anon"]
    user_id = resp1.json()["user_id"]

    # Logout
    resp2 = client.get("/logout", cookies={"deriva_chatbot_anon": anon_id})
    assert resp2.status_code == 302
    # Session removed from store
    from deriva_mcp_ui.auth import user_session_key
    assert await store.get(user_session_key(user_id)) is None


def test_config_no_credenza_url_skips_credenza_validation():
    """No credenza_url: validate_for_http() does not require Credenza fields."""
    s = Settings(
        mcp_url="http://mcp:8000",
        llm_api_key="sk-test",
    )
    s.validate_for_http()  # must not raise


def test_config_with_credenza_url_requires_full_config():
    """credenza_url set: validate_for_http() raises when other Credenza fields are missing."""
    s = Settings(
        mcp_url="http://mcp:8000",
        llm_api_key="sk-test",
        credenza_url="https://credenza.example.org",
    )
    with pytest.raises(ValueError, match="DERIVA_CHATBOT_CLIENT_ID"):
        s.validate_for_http()


@pytest.mark.asyncio
async def test_logout_falls_back_to_public_url_on_credenza_failure(app_and_store):
    """If Credenza /logout call fails, browser is redirected to public_url/."""
    app, store = app_and_store

    bearer = "tok-credenza-fail"
    now = time.time()
    await store.set(user_session_key("dave"), Session(user_id="dave", bearer_token=bearer, created_at=now, last_active=now))
    await store.set(_token_key(bearer), Session(user_id="dave", bearer_token=bearer, created_at=now, last_active=now))

    from unittest.mock import AsyncMock, patch

    with patch("deriva_mcp_ui.auth.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=Exception("credenza down"))
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = TestClient(app, follow_redirects=False)
        resp = client.get("/logout", cookies={"deriva_chatbot_session": bearer})

    assert resp.status_code == 302
    assert resp.headers["location"].endswith("/")
    assert await store.get(user_session_key("dave")) is None


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_endpoint_returns_messages(app_and_store):
    """GET /history returns display-friendly messages from the session."""
    app, store = app_and_store

    bearer = "tok-hist"
    now = time.time()
    session = Session(
        user_id="eve",
        bearer_token=bearer,
        created_at=now,
        last_active=now,
        history=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "get_schema", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "schema data"},
            {"role": "assistant", "content": "Here is the schema."},
        ],
    )
    await store.set(user_session_key("eve"), session)
    await store.set(_token_key(bearer), Session(user_id="eve", bearer_token=bearer, created_at=now, last_active=now))

    client = TestClient(app)
    resp = client.get("/history", cookies={"deriva_chatbot_session": bearer})
    assert resp.status_code == 200
    messages = resp.json()["messages"]

    # Should have: tool_use summary, assistant text
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1] == {"role": "tool_use", "tools": ["get_schema"]}
    assert messages[2] == {"role": "assistant", "content": "Here is the schema."}
    assert len(messages) == 3


@pytest.mark.asyncio
async def test_clear_history(app_and_store):
    """DELETE /history clears session history and history store entry."""
    app, store = app_and_store

    bearer = "tok-clear"
    now = time.time()
    session = Session(
        user_id="hank",
        bearer_token=bearer,
        created_at=now,
        last_active=now,
        history=[{"role": "user", "content": "hello"}],
    )
    await store.set(user_session_key("hank"), session)
    await store.set(_token_key(bearer), Session(user_id="hank", bearer_token=bearer, created_at=now, last_active=now))
    await store.set(history_key("hank"), Session(user_id="hank", history=session.history), ttl=604800)

    client = TestClient(app)
    resp = client.delete("/history", cookies={"deriva_chatbot_session": bearer})
    assert resp.status_code == 200

    # Session history should be empty
    updated = await store.get(user_session_key("hank"))
    assert updated is not None
    assert updated.history == []

    # History store entry should be gone
    assert await store.get(history_key("hank")) is None


@pytest.mark.asyncio
async def test_history_endpoint_empty(app_and_store):
    """GET /history returns empty list for a session with no history."""
    app, store = app_and_store

    bearer = "tok-empty-hist"
    now = time.time()
    session = Session(user_id="frank", bearer_token=bearer, created_at=now, last_active=now)
    await store.set(user_session_key("frank"), session)
    await store.set(_token_key(bearer), Session(user_id="frank", bearer_token=bearer, created_at=now, last_active=now))

    client = TestClient(app)
    resp = client.get("/history", cookies={"deriva_chatbot_session": bearer})
    assert resp.status_code == 200
    assert resp.json()["messages"] == []


@pytest.mark.asyncio
async def test_history_restored_on_reauth(app_and_store):
    """When uid: session is gone but history: entry exists, history is restored on login."""
    app, store = app_and_store

    # Simulate: history entry persists but session has expired
    saved_history = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
    ]
    hist_session = Session(user_id="grace", history=saved_history)
    await store.set(history_key("grace"), hist_session, ttl=604800)

    # No uid: entry for grace -- simulates expired session
    assert await store.get(user_session_key("grace")) is None

    # Simulate login callback
    with _mock_http(
        token_resp_json={"access_token": "new-tok-grace"},
        session_resp_json={"client": {"id": "grace", "display_name": "Grace"}},
    ):
        client = TestClient(app, follow_redirects=False)
        resp = client.get("/login")
        from urllib.parse import parse_qs, urlparse

        location = resp.headers["location"]
        query = parse_qs(urlparse(location).query)
        state_val = query["state"][0]

        client.get(
            f"/callback?code=abc&state={state_val}",
            cookies={"deriva_chatbot_pkce": resp.cookies["deriva_chatbot_pkce"]},
        )

    # After login, the session should have the restored history
    session = await store.get(user_session_key("grace"))
    assert session is not None
    assert len(session.history) == 2
    assert session.history[0]["content"] == "previous question"


@pytest.mark.asyncio
async def test_anonymous_history_restored_on_session_expiry(anon_app_and_store):
    """Anonymous session expired but cookie + history: entry exist -- history restored."""
    app, store = anon_app_and_store
    client = TestClient(app, follow_redirects=False)

    # Create initial anonymous session
    resp1 = client.get("/session-info")
    anon_id = resp1.cookies["deriva_chatbot_anon"]
    user_id = resp1.json()["user_id"]

    # Store history and then expire the session
    saved_history = [
        {"role": "user", "content": "anon question"},
        {"role": "assistant", "content": [{"type": "text", "text": "anon answer"}]},
    ]
    hist_session = Session(user_id=user_id, history=saved_history)
    await store.set(history_key(user_id), hist_session, ttl=604800)
    # Delete the uid: session to simulate expiry
    await store.delete(user_session_key(user_id))

    # Request with same cookie -- should get a new session with restored history
    resp2 = client.get("/session-info", cookies={"deriva_chatbot_anon": anon_id})
    assert resp2.status_code == 200
    assert resp2.json()["user_id"] == user_id

    # Verify history was restored
    session = await store.get(user_session_key(user_id))
    assert session is not None
    assert len(session.history) == 2
    assert session.history[0]["content"] == "anon question"
