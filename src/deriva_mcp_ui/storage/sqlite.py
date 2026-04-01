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


class SQLiteSessionStore:
    """Session store backed by SQLite via aiosqlite."""

    def __init__(self, url: str, ttl: int = 28800) -> None:
        try:
            import aiosqlite  # noqa: F401
        except ImportError as exc:
            raise ImportError("sqlite extra required: pip install 'deriva-mcp-ui[sqlite]'") from exc
        # Strip sqlite:/// scheme prefix if present
        self._path = url.removeprefix("sqlite:///")
        self._ttl = ttl
        self._db = None

    async def _init_db(self):
        if self._db is None:
            import aiosqlite

            self._db = await aiosqlite.connect(self._path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute(_CREATE_TABLE)
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

    async def set(self, session_id: str, session: Session) -> None:
        db = await self._init_db()
        await db.execute(_SQL_UPSERT, (session_id, session.to_json(), time.time() + self._ttl))
        await db.commit()

    async def delete(self, session_id: str) -> None:
        db = await self._init_db()
        await db.execute(_SQL_DELETE, (session_id,))
        await db.commit()

    async def sweep(self) -> None:
        db = await self._init_db()
        await db.execute(_SQL_SWEEP, (time.time(),))
        await db.commit()
