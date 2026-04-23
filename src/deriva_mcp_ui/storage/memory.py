"""In-process memory session store.

Suitable for development and single-worker deployments only.
State is lost on restart and not shared across processes.
"""

from __future__ import annotations

import asyncio
import time

from .base import Session


class MemorySessionStore:
    """TTL-aware in-process session store backed by a plain dict."""

    def __init__(self, ttl: int = 28800) -> None:
        self._ttl = ttl
        self._sessions: dict[str, tuple[Session, float]] = {}  # id -> (session, expires_at)
        self._user_costs: dict[str, dict] = {}  # user_id -> cost/token totals (no TTL)
        self._user_identities: dict[str, dict] = {}  # user_id -> identity record (no TTL)
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            session, expires_at = entry
            if time.monotonic() > expires_at:
                del self._sessions[session_id]
                return None
            return session

    async def set(self, session_id: str, session: Session, ttl: int | None = None) -> None:
        async with self._lock:
            self._sessions[session_id] = (session, time.monotonic() + (ttl or self._ttl))

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def sweep(self) -> None:
        now = time.monotonic()
        async with self._lock:
            expired = [sid for sid, (_, exp) in self._sessions.items() if now > exp]
            for sid in expired:
                del self._sessions[sid]

    async def increment_user_cost(
        self,
        user_id: str,
        cost_usd: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        async with self._lock:
            rec = self._user_costs.get(user_id)
            if rec is None:
                self._user_costs[user_id] = {
                    "lifetime_cost_usd": cost_usd,
                    "total_prompt_tokens": prompt_tokens,
                    "total_completion_tokens": completion_tokens,
                    "total_cache_read_tokens": cache_read_tokens,
                    "total_cache_creation_tokens": cache_creation_tokens,
                }
            else:
                rec["lifetime_cost_usd"] += cost_usd
                rec["total_prompt_tokens"] += prompt_tokens
                rec["total_completion_tokens"] += completion_tokens
                rec["total_cache_read_tokens"] += cache_read_tokens
                rec["total_cache_creation_tokens"] += cache_creation_tokens

    async def get_user_lifetime_cost(self, user_id: str) -> float:
        async with self._lock:
            rec = self._user_costs.get(user_id)
            return rec["lifetime_cost_usd"] if rec is not None else 0.0

    async def upsert_user_identity(self, user_id: str, email: str, full_name: str) -> None:
        now = time.time()
        async with self._lock:
            existing = self._user_identities.get(user_id)
            if existing is None:
                self._user_identities[user_id] = {
                    "user_id": user_id,
                    "email": email,
                    "full_name": full_name,
                    "first_seen": now,
                    "last_seen": now,
                }
            else:
                existing["email"] = email
                existing["full_name"] = full_name
                existing["last_seen"] = now

    async def get_user_last_seen(self, user_id: str) -> float | None:
        async with self._lock:
            rec = self._user_identities.get(user_id)
            return rec["last_seen"] if rec is not None else None
