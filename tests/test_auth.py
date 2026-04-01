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
        anthropic_api_key="sk-ant-test",
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

    from unittest.mock import AsyncMock, patch

    with patch("deriva_mcp_ui.auth.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = TestClient(app, follow_redirects=False)
        resp = client.get("/logout", cookies={"deriva_chatbot_session": bearer})

    assert resp.status_code == 302
    mock_http.post.assert_called_once()
    assert mock_http.post.call_args.kwargs["data"]["token"] == bearer

    assert await store.get(user_session_key("alice")) is None
    assert await store.get(_token_key(bearer)) is None


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


@pytest.mark.asyncio
async def test_logout_revocation_failure_is_nonfatal(app_and_store):
    """Token revocation HTTP error is logged but logout still completes (lines 292-293)."""
    app, store = app_and_store

    bearer = "tok-revoke-fail"
    now = time.time()
    session = Session(user_id="dave", bearer_token=bearer, created_at=now, last_active=now)
    await store.set(user_session_key("dave"), session)
    await store.set(
        _token_key(bearer),
        Session(user_id="dave", bearer_token=bearer, created_at=now, last_active=now),
    )

    from unittest.mock import AsyncMock, patch

    async def _raise(*a, **kw):
        raise Exception("revocation server down")

    with patch("deriva_mcp_ui.auth.httpx.AsyncClient") as mock_cls:
        mock_http = MagicMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_http.post = _raise

        client = TestClient(app, follow_redirects=False)
        resp = client.get("/logout", cookies={"deriva_chatbot_session": bearer})

    # Logout must succeed despite revocation failure
    assert resp.status_code == 302
    assert await store.get(user_session_key("dave")) is None
