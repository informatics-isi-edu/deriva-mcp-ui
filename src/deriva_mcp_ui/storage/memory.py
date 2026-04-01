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

    async def set(self, session_id: str, session: Session) -> None:
        async with self._lock:
            self._sessions[session_id] = (session, time.monotonic() + self._ttl)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def sweep(self) -> None:
        now = time.monotonic()
        async with self._lock:
            expired = [sid for sid, (_, exp) in self._sessions.items() if now > exp]
            for sid in expired:
                del self._sessions[sid]
