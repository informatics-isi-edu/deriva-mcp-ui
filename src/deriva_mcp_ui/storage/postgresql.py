"""PostgreSQL session store backend.

Uses asyncpg with per-connection prepared statements. The pool init callback runs DDL (idempotent) and
prepares all statements on each new connection so query planning is done once
per connection rather than on every call.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from .base import Session

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chatbot_sessions (
    session_id TEXT        PRIMARY KEY,
    data       TEXT        NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
)
"""

# Prepared statement SQL -- referenced by name in comments below
_SQL_GET = "SELECT data FROM chatbot_sessions WHERE session_id = $1 AND expires_at > NOW()"

_SQL_UPSERT = """
INSERT INTO chatbot_sessions (session_id, data, expires_at)
VALUES ($1, $2, $3)
ON CONFLICT (session_id) DO UPDATE
    SET data = EXCLUDED.data, expires_at = EXCLUDED.expires_at
"""

_SQL_DELETE = "DELETE FROM chatbot_sessions WHERE session_id = $1"

_SQL_SWEEP = "DELETE FROM chatbot_sessions WHERE expires_at <= NOW()"


async def _init_conn(conn) -> None:
    """Called by asyncpg for each new pool connection.

    Runs the schema DDL once per connection (idempotent) then prepares all statements
    so the query plan is cached at the wire-protocol level for that connection.
    """
    await conn.execute(_CREATE_TABLE)
    # Store PreparedStatement objects as connection attributes so callers can
    # invoke them without re-preparing on every request.
    conn._stmt_get = await conn.prepare(_SQL_GET)
    conn._stmt_upsert = await conn.prepare(_SQL_UPSERT)
    conn._stmt_delete = await conn.prepare(_SQL_DELETE)
    conn._stmt_sweep = await conn.prepare(_SQL_SWEEP)
    logger.debug("PostgreSQL connection initialized with prepared statements")


class PostgreSQLSessionStore:
    """Session store backed by PostgreSQL via asyncpg."""

    def __init__(self, url: str, ttl: int = 28800) -> None:
        try:
            import asyncpg  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "postgresql extra required: pip install 'deriva-mcp-ui[postgresql]'"
            ) from exc
        self._url = url
        self._ttl = ttl
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self._url, init=_init_conn)
        return self._pool

    async def get(self, session_id: str) -> Session | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn._stmt_get.fetchrow(session_id)
        if row is None:
            return None
        return Session.from_json(row["data"])

    async def set(self, session_id: str, session: Session) -> None:
        expires_at = datetime.now(UTC) + timedelta(seconds=self._ttl)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn._stmt_upsert.execute(session_id, session.to_json(), expires_at)

    async def delete(self, session_id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn._stmt_delete.execute(session_id)

    async def sweep(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn._stmt_sweep.execute()
