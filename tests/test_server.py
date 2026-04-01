"""Tests for server.py routes: /health, / (index)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from deriva_mcp_ui.auth import _token_key, user_session_key
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
        anthropic_api_key="sk-ant-test",
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
