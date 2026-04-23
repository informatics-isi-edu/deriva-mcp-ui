"""SQLite session store backend.

Uses aiosqlite. SQLite does not expose a named PREPARE/EXECUTE API; instead,
sqlite3 compiles and caches bytecode for each unique parameterized query string
automatically. We use the same `?`-parameterized SQL string on every call,
schema creation runs once in _init_db().
"""

from __future__ import annotations

import logging
import time

from .base import Session

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chatbot_sessions (
    session_id TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    expires_at REAL NOT NULL
)
"""

_CREATE_COSTS_TABLE = """
CREATE TABLE IF NOT EXISTS chatbot_user_costs (
    user_id                  TEXT PRIMARY KEY,
    lifetime_cost_usd        REAL    NOT NULL DEFAULT 0,
    total_prompt_tokens      INTEGER NOT NULL DEFAULT 0,
    total_completion_tokens  INTEGER NOT NULL DEFAULT 0,
    total_cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
    total_cache_creation_tokens INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS chatbot_users (
    user_id    TEXT    PRIMARY KEY,
    email      TEXT    NOT NULL DEFAULT '',
    full_name  TEXT    NOT NULL DEFAULT '',
    first_seen REAL    NOT NULL,
    last_seen  REAL    NOT NULL
)
"""

# These SQL strings are module-level constants so the same string object
# (and therefore the same sqlite3 compiled bytecode entry) is used on every
# call -- the closest SQLite equivalent to named prepared statements.
_SQL_GET = "SELECT data FROM chatbot_sessions WHERE session_id = ? AND expires_at > ?"
_SQL_UPSERT = """
INSERT INTO chatbot_sessions (session_id, data, expires_at)
VALUES (?, ?, ?)
ON CONFLICT(session_id) DO UPDATE
    SET data = excluded.data, expires_at = excluded.expires_at
"""
_SQL_DELETE = "DELETE FROM chatbot_sessions WHERE session_id = ?"
_SQL_SWEEP = "DELETE FROM chatbot_sessions WHERE expires_at <= ?"

_SQL_INCREMENT_COST = """
INSERT INTO chatbot_user_costs
    (user_id, lifetime_cost_usd, total_prompt_tokens, total_completion_tokens,
     total_cache_read_tokens, total_cache_creation_tokens)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(user_id) DO UPDATE
    SET lifetime_cost_usd        = lifetime_cost_usd        + excluded.lifetime_cost_usd,
        total_prompt_tokens      = total_prompt_tokens      + excluded.total_prompt_tokens,
        total_completion_tokens  = total_completion_tokens  + excluded.total_completion_tokens,
        total_cache_read_tokens  = total_cache_read_tokens  + excluded.total_cache_read_tokens,
        total_cache_creation_tokens = total_cache_creation_tokens + excluded.total_cache_creation_tokens
"""
_SQL_GET_COST = "SELECT lifetime_cost_usd FROM chatbot_user_costs WHERE user_id = ?"

_SQL_GET_USER_LAST_SEEN = "SELECT last_seen FROM chatbot_users WHERE user_id = ?"

_SQL_UPSERT_USER = """
INSERT INTO chatbot_users (user_id, email, full_name, first_seen, last_seen)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(user_id) DO UPDATE
    SET email = excluded.email,
        full_name = excluded.full_name,
        last_seen = excluded.last_seen
"""


class SQLiteSessionStore:
    """Session store backed by SQLite via aiosqlite."""

    def __init__(self, url: str, ttl: int = 28800) -> None:
        try:
            import aiosqlite  # noqa: F401
        except ImportError as exc:
            raise ImportError("sqlite extra required: pip install 'deriva-mcp-ui[sqlite]'") from exc
        # Strip sqlite: scheme prefix if present, preserving absolute paths.
        # Accepts sqlite:///path (relative) and sqlite:////path (absolute)
        # as well as plain file paths.
        if url.startswith("sqlite://"):
            self._path = url[len("sqlite://"):]
        else:
            self._path = url
        self._ttl = ttl
        self._db = None

    async def _init_db(self):
        if self._db is None:
            import aiosqlite

            self._db = await aiosqlite.connect(self._path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute(_CREATE_TABLE)
            await self._db.execute(_CREATE_COSTS_TABLE)
            await self._db.execute(_CREATE_USERS_TABLE)
            await self._db.commit()
            logger.debug("SQLite session store initialized at %s", self._path)
        return self._db

    async def get(self, session_id: str) -> Session | None:
        db = await self._init_db()
        async with db.execute(_SQL_GET, (session_id, time.time())) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return Session.from_json(row[0])

    async def set(self, session_id: str, session: Session, ttl: int | None = None) -> None:
        db = await self._init_db()
        await db.execute(_SQL_UPSERT, (session_id, session.to_json(), time.time() + (ttl or self._ttl)))
        await db.commit()

    async def delete(self, session_id: str) -> None:
        db = await self._init_db()
        await db.execute(_SQL_DELETE, (session_id,))
        await db.commit()

    async def sweep(self) -> None:
        db = await self._init_db()
        await db.execute(_SQL_SWEEP, (time.time(),))
        await db.commit()

    async def increment_user_cost(
        self,
        user_id: str,
        cost_usd: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        db = await self._init_db()
        await db.execute(
            _SQL_INCREMENT_COST,
            (user_id, cost_usd, prompt_tokens, completion_tokens, cache_read_tokens, cache_creation_tokens),
        )
        await db.commit()

    async def get_user_lifetime_cost(self, user_id: str) -> float:
        db = await self._init_db()
        async with db.execute(_SQL_GET_COST, (user_id,)) as cursor:
            row = await cursor.fetchone()
        return row[0] if row is not None else 0.0

    async def upsert_user_identity(self, user_id: str, email: str, full_name: str) -> None:
        now = time.time()
        db = await self._init_db()
        await db.execute(_SQL_UPSERT_USER, (user_id, email, full_name, now, now))
        await db.commit()

    async def get_user_last_seen(self, user_id: str) -> float | None:
        db = await self._init_db()
        async with db.execute(_SQL_GET_USER_LAST_SEEN, (user_id,)) as cursor:
            row = await cursor.fetchone()
        return row[0] if row is not None else None
