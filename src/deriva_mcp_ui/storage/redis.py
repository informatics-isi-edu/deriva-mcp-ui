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

    async def increment_user_cost(
        self,
        user_id: str,
        cost_usd: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        # All INCRBYFLOAT/INCRBY calls are atomic; no TTL so values persist across sessions.
        key = f"user_cost:{user_id}"
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.incrbyfloat(key, cost_usd)
            pipe.hincrbyfloat(f"user_tokens:{user_id}", "prompt", prompt_tokens)
            pipe.hincrbyfloat(f"user_tokens:{user_id}", "completion", completion_tokens)
            pipe.hincrbyfloat(f"user_tokens:{user_id}", "cache_read", cache_read_tokens)
            pipe.hincrbyfloat(f"user_tokens:{user_id}", "cache_creation", cache_creation_tokens)
            await pipe.execute()

    async def get_user_lifetime_cost(self, user_id: str) -> float:
        val = await self._client.get(f"user_cost:{user_id}")
        return float(val) if val is not None else 0.0

    async def get_user_last_seen(self, user_id: str) -> float | None:
        val = await self._client.hget(f"user_identity:{user_id}", "last_seen")
        return float(val) if val is not None else None

    async def upsert_user_identity(self, user_id: str, email: str, full_name: str) -> None:
        import time
        key = f"user_identity:{user_id}"
        now = str(time.time())
        # HSETNX sets first_seen only on first call; subsequent calls skip it.
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.hsetnx(key, "first_seen", now)
            pipe.hset(key, mapping={"email": email, "full_name": full_name, "user_id": user_id, "last_seen": now})
            await pipe.execute()
