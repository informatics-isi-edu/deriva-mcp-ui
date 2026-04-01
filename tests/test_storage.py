"""Tests for the storage layer."""

import time

import pytest

from deriva_mcp_ui.storage import Session, create_store
from deriva_mcp_ui.storage.memory import MemorySessionStore


def _make_session(user_id: str = "alice") -> Session:
    now = time.time()
    return Session(user_id=user_id, bearer_token="tok", created_at=now, last_active=now)


# ---------------------------------------------------------------------------
# Session serialization
# ---------------------------------------------------------------------------


def test_session_roundtrip():
    s = _make_session()
    s.history = [{"role": "user", "content": "hello"}]
    s2 = Session.from_json(s.to_json())
    assert s2.user_id == s.user_id
    assert s2.bearer_token == s.bearer_token
    assert s2.history == s.history
    assert s2.tools is None
    assert s2.schema_primed is False


# ---------------------------------------------------------------------------
# MemorySessionStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_set_get():
    store = MemorySessionStore(ttl=60)
    session = _make_session()
    await store.set("sid1", session)
    result = await store.get("sid1")
    assert result is not None
    assert result.user_id == "alice"


@pytest.mark.asyncio
async def test_memory_missing_returns_none():
    store = MemorySessionStore(ttl=60)
    assert await store.get("nonexistent") is None


@pytest.mark.asyncio
async def test_memory_delete():
    store = MemorySessionStore(ttl=60)
    await store.set("sid1", _make_session())
    await store.delete("sid1")
    assert await store.get("sid1") is None


@pytest.mark.asyncio
async def test_memory_expiry():
    store = MemorySessionStore(ttl=1)
    await store.set("sid1", _make_session())
    # Manually force expiry
    store._sessions["sid1"] = (store._sessions["sid1"][0], 0.0)
    assert await store.get("sid1") is None


@pytest.mark.asyncio
async def test_memory_sweep():
    store = MemorySessionStore(ttl=60)
    await store.set("sid1", _make_session("alice"))
    await store.set("sid2", _make_session("bob"))
    # Force sid2 to expire
    store._sessions["sid2"] = (store._sessions["sid2"][0], 0.0)
    await store.sweep()
    assert "sid1" in store._sessions
    assert "sid2" not in store._sessions


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_create_store_memory():
    store = create_store("memory", "", ttl=60)
    assert isinstance(store, MemorySessionStore)


def test_create_store_unknown():
    with pytest.raises(ValueError, match="Unknown storage backend"):
        create_store("bogus", "url")


def test_create_store_missing_url():
    with pytest.raises(ValueError, match="DERIVA_CHATBOT_STORAGE_BACKEND_URL"):
        create_store("redis", "")
