"""Multi-backend contract tests for the SessionStore implementations.

The same CRUD + expiry tests run against every backend via a single
parameterized fixture, following the pattern in
credenza/test/api/session/test_storage_backend.py and
credenza/test/api/session/test_session_store.py.

Redis and Valkey are exercised via fakeredis.aioredis so no real server is
needed.  PostgreSQL uses testing.postgresql (skipped on Windows).  SQLite uses
an in-memory database.
"""

from __future__ import annotations

import platform
import time
import uuid

import pytest

from deriva_mcp_ui.storage.base import Session
from deriva_mcp_ui.storage.memory import MemorySessionStore
from deriva_mcp_ui.storage.postgresql import PostgreSQLSessionStore
from deriva_mcp_ui.storage.redis import RedisSessionStore
from deriva_mcp_ui.storage.sqlite import SQLiteSessionStore
from deriva_mcp_ui.storage.valkey import ValkeySessionStore

# ---------------------------------------------------------------------------
# Module-level PostgreSQL instance (created once, skipped on Windows)
# ---------------------------------------------------------------------------

_postgresql = None
if platform.system() != "Windows":
    import testing.postgresql

    _postgresql = testing.postgresql.Postgresql()


# ---------------------------------------------------------------------------
# Parameterized store fixture
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=["memory", "redis", "valkey", "sqlite", "postgresql"],
    ids=lambda n: n,
    scope="function",
)
async def store(request, monkeypatch):
    """Yield a fully initialised SessionStore for each backend.

    Mirrors the fixture structure in Credenza's test_storage_backend.py:
    one fixture, all backends, no test duplication.
    """
    param = request.param

    if param == "memory":
        yield MemorySessionStore(ttl=60)

    elif param in ("redis", "valkey"):
        import fakeredis.aioredis
        import redis.asyncio

        # FakeServer gives each fixture its own isolated keyspace
        server = fakeredis.FakeServer()
        fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        # Patch redis.asyncio.from_url so both RedisSessionStore and
        # ValkeySessionStore (which inherits it) receive the fake client
        monkeypatch.setattr(redis.asyncio, "from_url", lambda url, **kw: fake)
        cls = RedisSessionStore if param == "redis" else ValkeySessionStore
        yield cls(url=f"{param}://fake", ttl=60)

    elif param == "sqlite":
        s = SQLiteSessionStore(url=":memory:", ttl=60)
        yield s
        if s._db is not None:
            await s._db.close()

    elif param == "postgresql":
        if platform.system() == "Windows":
            pytest.skip("PostgreSQL backend tests skipped on Windows")
        s = PostgreSQLSessionStore(url=_postgresql.url(), ttl=60)
        yield s
        if s._pool is not None:
            await s._pool.close()

    else:
        raise RuntimeError(f"Unknown backend param: {param}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(user_id: str = "alice") -> Session:
    now = time.time()
    return Session(user_id=user_id, bearer_token=f"tok-{user_id}", created_at=now, last_active=now)


def _sid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Contract tests -- run against every backend
# ---------------------------------------------------------------------------


async def test_set_and_get(store):
    sid = _sid()
    await store.set(sid, _session("alice"))
    result = await store.get(sid)
    assert result is not None
    assert result.user_id == "alice"
    assert result.bearer_token == "tok-alice"


async def test_get_missing_returns_none(store):
    assert await store.get(_sid()) is None


async def test_delete_removes_session(store):
    sid = _sid()
    await store.set(sid, _session())
    await store.delete(sid)
    assert await store.get(sid) is None


async def test_delete_nonexistent_is_silent(store):
    # Must not raise
    await store.delete(_sid())


async def test_overwrite_updates_session(store):
    sid = _sid()
    await store.set(sid, _session("alice"))

    updated = _session("alice")
    updated.bearer_token = "new-tok"
    updated.history = [{"role": "user", "content": "hello"}]
    await store.set(sid, updated)

    result = await store.get(sid)
    assert result is not None
    assert result.bearer_token == "new-tok"
    assert result.history == [{"role": "user", "content": "hello"}]


async def test_tools_and_flags_roundtrip(store):
    sid = _sid()
    s = _session("bob")
    s.tools = [{"name": "get_entities", "description": "...", "input_schema": {}}]
    s.schema_primed = True
    s.gp_hostname = "example.org"
    s.gp_catalog_id = "1"
    await store.set(sid, s)

    result = await store.get(sid)
    assert result is not None
    assert result.tools is not None
    assert result.tools[0]["name"] == "get_entities"
    assert result.schema_primed is True
    assert result.gp_hostname == "example.org"
    assert result.gp_catalog_id == "1"


async def test_multiple_sessions_isolated(store):
    sid1, sid2 = _sid(), _sid()
    await store.set(sid1, _session("alice"))
    await store.set(sid2, _session("bob"))

    r1 = await store.get(sid1)
    r2 = await store.get(sid2)
    assert r1 is not None and r1.user_id == "alice"
    assert r2 is not None and r2.user_id == "bob"

    await store.delete(sid1)
    assert await store.get(sid1) is None
    assert await store.get(sid2) is not None


# ---------------------------------------------------------------------------
# Expiry tests -- memory and sqlite only (redis/valkey TTL is hardware-timed;
# postgresql uses server-side NOW() which we cannot monkeypatch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_name", ["memory", "sqlite"])
async def test_expired_session_not_returned(request, monkeypatch, backend_name):
    """A session whose TTL has passed must be treated as absent."""
    # Re-use the store fixture indirectly by creating the store directly
    # so we can control the TTL and time without fixture indirection.
    if backend_name == "memory":
        s = MemorySessionStore(ttl=5)
        sid = _sid()
        await s.set(sid, _session())
        # Force expiry by backdating the stored entry
        entry = s._sessions[sid]
        s._sessions[sid] = (entry[0], time.monotonic() - 1)
        assert await s.get(sid) is None

    elif backend_name == "sqlite":
        s = SQLiteSessionStore(url=":memory:", ttl=5)
        sid = _sid()
        await s.set(sid, _session())
        # Backdate the stored expires_at directly in the DB
        db = await s._init_db()
        await db.execute(
            "UPDATE chatbot_sessions SET expires_at = ? WHERE session_id = ?",
            (time.time() - 1, sid),
        )
        await db.commit()
        assert await s.get(sid) is None
        await db.close()


async def test_sweep_removes_expired_memory(monkeypatch):
    """sweep() evicts expired entries from the memory store."""
    s = MemorySessionStore(ttl=60)
    sid_live = _sid()
    sid_dead = _sid()
    await s.set(sid_live, _session("live"))
    await s.set(sid_dead, _session("dead"))

    # Force sid_dead to expire
    entry = s._sessions[sid_dead]
    s._sessions[sid_dead] = (entry[0], time.monotonic() - 1)

    await s.sweep()

    assert sid_live in s._sessions
    assert sid_dead not in s._sessions


# ---------------------------------------------------------------------------
# create_store factory
# ---------------------------------------------------------------------------


def test_create_store_memory_no_url():
    from deriva_mcp_ui.storage import create_store

    store = create_store("memory", "", ttl=60)
    from deriva_mcp_ui.storage.memory import MemorySessionStore

    assert isinstance(store, MemorySessionStore)


def test_create_store_unknown_backend_raises():
    from deriva_mcp_ui.storage import create_store

    with pytest.raises(ValueError, match="Unknown storage backend"):
        create_store("bogus", "bogus://url")


def test_create_store_missing_url_raises():
    from deriva_mcp_ui.storage import create_store

    with pytest.raises(ValueError, match="STORAGE_BACKEND_URL"):
        create_store("sqlite", "")


def test_create_store_lazy_import_sqlite():
    """create_store hits the importlib lazy-import path for non-memory backends."""
    from deriva_mcp_ui.storage import create_store
    from deriva_mcp_ui.storage.sqlite import SQLiteSessionStore

    store = create_store("sqlite", "/tmp/test-deriva-mcp-ui.db", ttl=60)
    assert isinstance(store, SQLiteSessionStore)
