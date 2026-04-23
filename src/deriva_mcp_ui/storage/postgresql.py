"""PostgreSQL session store backend.

Uses asyncpg. The pool init callback runs DDL (idempotent) on each new
connection. Queries use asyncpg's built-in statement cache (enabled by
default) so the query plan is reused without manual statement preparation.
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

_CREATE_COSTS_TABLE = """
CREATE TABLE IF NOT EXISTS chatbot_user_costs (
    user_id                     TEXT             PRIMARY KEY,
    lifetime_cost_usd           DOUBLE PRECISION NOT NULL DEFAULT 0,
    total_prompt_tokens         BIGINT           NOT NULL DEFAULT 0,
    total_completion_tokens     BIGINT           NOT NULL DEFAULT 0,
    total_cache_read_tokens     BIGINT           NOT NULL DEFAULT 0,
    total_cache_creation_tokens BIGINT           NOT NULL DEFAULT 0
)
"""

_CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS chatbot_users (
    user_id    TEXT                     PRIMARY KEY,
    email      TEXT                     NOT NULL DEFAULT '',
    full_name  TEXT                     NOT NULL DEFAULT '',
    first_seen DOUBLE PRECISION         NOT NULL,
    last_seen  DOUBLE PRECISION         NOT NULL
)
"""

_SQL_GET = "SELECT data FROM chatbot_sessions WHERE session_id = $1 AND expires_at > NOW()"

_SQL_UPSERT = """
INSERT INTO chatbot_sessions (session_id, data, expires_at)
VALUES ($1, $2, $3)
ON CONFLICT (session_id) DO UPDATE
    SET data = EXCLUDED.data, expires_at = EXCLUDED.expires_at
"""

_SQL_DELETE = "DELETE FROM chatbot_sessions WHERE session_id = $1"

_SQL_SWEEP = "DELETE FROM chatbot_sessions WHERE expires_at <= NOW()"

_SQL_INCREMENT_COST = """
INSERT INTO chatbot_user_costs
    (user_id, lifetime_cost_usd, total_prompt_tokens, total_completion_tokens,
     total_cache_read_tokens, total_cache_creation_tokens)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (user_id) DO UPDATE
    SET lifetime_cost_usd           = chatbot_user_costs.lifetime_cost_usd           + EXCLUDED.lifetime_cost_usd,
        total_prompt_tokens         = chatbot_user_costs.total_prompt_tokens         + EXCLUDED.total_prompt_tokens,
        total_completion_tokens     = chatbot_user_costs.total_completion_tokens     + EXCLUDED.total_completion_tokens,
        total_cache_read_tokens     = chatbot_user_costs.total_cache_read_tokens     + EXCLUDED.total_cache_read_tokens,
        total_cache_creation_tokens = chatbot_user_costs.total_cache_creation_tokens + EXCLUDED.total_cache_creation_tokens
"""
_SQL_GET_COST = "SELECT lifetime_cost_usd FROM chatbot_user_costs WHERE user_id = $1"

_SQL_GET_USER_LAST_SEEN = "SELECT last_seen FROM chatbot_users WHERE user_id = $1"

_SQL_UPSERT_USER = """
INSERT INTO chatbot_users (user_id, email, full_name, first_seen, last_seen)
VALUES ($1, $2, $3, $4, $4)
ON CONFLICT (user_id) DO UPDATE
    SET email     = EXCLUDED.email,
        full_name = EXCLUDED.full_name,
        last_seen = EXCLUDED.last_seen
"""


async def _init_conn(conn) -> None:
    """Called by asyncpg for each new pool connection. Runs DDL (idempotent)."""
    await conn.execute(_CREATE_TABLE)
    await conn.execute(_CREATE_COSTS_TABLE)
    await conn.execute(_CREATE_USERS_TABLE)


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
            row = await conn.fetchrow(_SQL_GET, session_id)
        if row is None:
            return None
        return Session.from_json(row["data"])

    async def set(self, session_id: str, session: Session, ttl: int | None = None) -> None:
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl or self._ttl)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_SQL_UPSERT, session_id, session.to_json(), expires_at)

    async def delete(self, session_id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_SQL_DELETE, session_id)

    async def sweep(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_SQL_SWEEP)

    async def increment_user_cost(
        self,
        user_id: str,
        cost_usd: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                _SQL_INCREMENT_COST,
                user_id, cost_usd, prompt_tokens, completion_tokens,
                cache_read_tokens, cache_creation_tokens,
            )

    async def get_user_lifetime_cost(self, user_id: str) -> float:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_COST, user_id)
        return row["lifetime_cost_usd"] if row is not None else 0.0

    async def upsert_user_identity(self, user_id: str, email: str, full_name: str) -> None:
        import time
        now = time.time()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_SQL_UPSERT_USER, user_id, email, full_name, now)

    async def get_user_last_seen(self, user_id: str) -> float | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_USER_LAST_SEEN, user_id)
        return row["last_seen"] if row is not None else None
