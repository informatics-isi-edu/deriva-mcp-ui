"""Valkey session store backend.

Valkey is Redis-compatible; this backend reuses the redis.asyncio client.
"""

from __future__ import annotations

from .redis import RedisSessionStore


class ValkeySessionStore(RedisSessionStore):
    """Session store backed by Valkey (Redis-compatible)."""
