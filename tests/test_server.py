"""Tests for server.py routes: /health, / (index)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from deriva_mcp_ui.auth import _token_key, require_session, user_session_key
from deriva_mcp_ui.config import Settings
from deriva_mcp_ui.server import create_app
from deriva_mcp_ui.storage.base import Session
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


@pytest.fixture()
def app_and_store():
    settings = _test_settings()
    app = create_app(settings)
    store = MemorySessionStore(ttl=settings.session_ttl)
    app.state.store = store
    return app, store


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(app_and_store):
    app, _ = app_and_store
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# / (index)
# ---------------------------------------------------------------------------


def test_index_returns_json_when_no_static(app_and_store):
    """With no static/index.html built, the index route returns a JSON status."""
    app, _ = app_and_store
    client = TestClient(app)
    resp = client.get("/")
    # Either 200 JSON (no static file) or FileResponse (static file present)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /chat -- error paths
# ---------------------------------------------------------------------------


async def test_chat_message_too_long(app_and_store):
    """Messages exceeding max_message_length return 400."""
    app, store = app_and_store

    bearer = "tok-len-test"
    now = time.time()
    session = Session(user_id="alice", bearer_token=bearer, created_at=now, last_active=now)
    await store.set(user_session_key("alice"), session)
    await store.set(
        _token_key(bearer),
        Session(user_id="alice", bearer_token=bearer, created_at=now, last_active=now),
    )

    client = TestClient(app)
    long_msg = "x" * 10001
    resp = client.post(
        "/chat",
        json={"message": long_msg},
        cookies={"deriva_chatbot_session": bearer},
    )
    assert resp.status_code == 400
    assert "maximum length" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /session-info
# ---------------------------------------------------------------------------


def test_session_info_includes_identity_fields():
    settings = _test_settings(default_hostname="example.org", default_catalog_id="1")
    app = create_app(settings)
    store = MemorySessionStore(ttl=settings.session_ttl)
    app.state.store = store

    now = time.time()
    session = Session(
        user_id="https://idp.example.org/sub123",
        bearer_token="tok",
        credenza_session={
            "preferred_username": "jdoe",
            "full_name": "Jane Doe",
            "email": "jane@example.org",
        },
        created_at=now,
        last_active=now,
    )
    app.dependency_overrides[require_session] = lambda: session

    client = TestClient(app)
    resp = client.get("/session-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["display_name"] == "jdoe"
    assert data["full_name"] == "Jane Doe"
    assert data["email"] == "jane@example.org"
    assert data["hostname"] == "example.org"
    assert data["user_id"] == "https://idp.example.org/sub123"


def test_session_info_anonymous_mode():
    settings = Settings(mcp_url="http://mcp", llm_api_key="sk-test")
    app = create_app(settings)
    store = MemorySessionStore(ttl=settings.session_ttl)
    app.state.store = store

    now = time.time()
    session = Session(user_id="anonymous/abc", created_at=now, last_active=now)
    app.dependency_overrides[require_session] = lambda: session

    client = TestClient(app)
    resp = client.get("/session-info")
    data = resp.json()
    assert data["display_name"] == "Anonymous"
    assert data["full_name"] == ""
    assert data["email"] == ""
    assert data["hostname"] == ""


def test_session_info_includes_operating_mode():
    settings = _test_settings()
    app = create_app(settings)
    store = MemorySessionStore(ttl=settings.session_ttl)
    app.state.store = store

    now = time.time()
    session = Session(user_id="alice", bearer_token="tok", created_at=now, last_active=now)
    app.dependency_overrides[require_session] = lambda: session

    client = TestClient(app)
    resp = client.get("/session-info")
    data = resp.json()
    assert data["operating_mode"] == "llm"


def test_session_info_operating_mode_rag_only():
    settings = Settings(mcp_url="http://mcp", mode="rag_only")
    app = create_app(settings)
    store = MemorySessionStore(ttl=settings.session_ttl)
    app.state.store = store

    now = time.time()
    session = Session(user_id="anon", created_at=now, last_active=now)
    app.dependency_overrides[require_session] = lambda: session

    client = TestClient(app)
    resp = client.get("/session-info")
    data = resp.json()
    assert data["operating_mode"] == "rag_only"


# ---------------------------------------------------------------------------
# /history -- GET and DELETE
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_with_session():
    """App with dependency override so auth is bypassed."""
    settings = _test_settings()
    app = create_app(settings)
    store = MemorySessionStore(ttl=settings.session_ttl)
    app.state.store = store

    now = time.time()
    session = Session(
        user_id="test-user",
        bearer_token="tok",
        created_at=now,
        last_active=now,
    )
    app.dependency_overrides[require_session] = lambda: session
    return app, store, session


def test_get_history_empty(app_with_session):
    app, _, _ = app_with_session
    client = TestClient(app)
    resp = client.get("/history")
    assert resp.status_code == 200
    assert resp.json() == {"messages": [], "input_history": []}


def test_get_history_extracts_messages(app_with_session):
    """History endpoint extracts text and tool calls from OpenAI-format messages."""
    app, _, session = app_with_session
    session.history = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "get_schema", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "tc1", "content": "schema data"},
        {"role": "assistant", "content": "Here is the schema."},
        # tool_result user messages (non-string content) should be skipped
        {"role": "user", "content": [{"type": "tool_result", "content": "..."}]},
    ]
    client = TestClient(app)
    resp = client.get("/history")
    messages = resp.json()["messages"]
    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1] == {"role": "tool_use", "tools": ["get_schema"]}
    assert messages[2] == {"role": "assistant", "content": "Here is the schema."}


def test_clear_history(app_with_session):
    app, _, session = app_with_session
    session.history = [{"role": "user", "content": "old msg"}]
    session.tools = [{"name": "tool1"}]
    session.schema_primed = True
    session.primed_schema = "schema text"
    session.primed_guides = "guide text"
    session.primed_ermrest = "ermrest text"

    client = TestClient(app)
    resp = client.delete("/history")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert session.history == []
    assert session.tools is None
    assert session.schema_primed is False
    assert session.primed_schema == ""
    assert session.primed_guides == ""
    assert session.primed_ermrest == ""
