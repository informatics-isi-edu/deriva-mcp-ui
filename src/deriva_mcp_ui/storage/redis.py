"""Redis session store backend."""

from __future__ import annotations

from .base import Session


class RedisSessionStore:
    """Session store backed by Redis."""

    def __init__(self, url: str, ttl: int = 28800) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:  # pragma: no cover
            raise ImportError("redis extra required: pip install 'deriva-mcp-ui[redis]'") from exc
        self._client = aioredis.from_url(url, decode_responses=True)
        self._ttl = ttl

    async def get(self, session_id: str) -> Session | None:
        data = await self._client.get(f"session:{session_id}")
        if data is None:
            return None
        return Session.from_json(data)

    async def set(self, session_id: str, session: Session, ttl: int | None = None) -> None:
        await self._client.setex(f"session:{session_id}", ttl or self._ttl, session.to_json())

    async def delete(self, session_id: str) -> None:
        await self._client.delete(f"session:{session_id}")

    async def sweep(self) -> None:
        pass  # Redis TTL handles expiry automatically  # pragma: no cover
