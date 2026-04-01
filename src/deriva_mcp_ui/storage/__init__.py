"""Session store backends and factory."""

from __future__ import annotations

from .base import Session, SessionStore
from .memory import MemorySessionStore

STORAGE_BACKENDS = {
    "memory": "deriva_mcp_ui.storage.memory:MemorySessionStore",
    "redis": "deriva_mcp_ui.storage.redis:RedisSessionStore",
    "valkey": "deriva_mcp_ui.storage.valkey:ValkeySessionStore",
    "postgresql": "deriva_mcp_ui.storage.postgresql:PostgreSQLSessionStore",
    "sqlite": "deriva_mcp_ui.storage.sqlite:SQLiteSessionStore",
}


def create_store(backend: str, url: str, ttl: int = 28800) -> SessionStore:
    """Return a session store instance for the given backend name."""
    if backend == "memory":
        return MemorySessionStore(ttl=ttl)

    if backend not in STORAGE_BACKENDS:
        raise ValueError(
            f"Unknown storage backend {backend!r}. Valid options: {', '.join(STORAGE_BACKENDS)}"
        )

    if not url:
        raise ValueError(f"DERIVA_CHATBOT_STORAGE_BACKEND_URL is required for backend {backend!r}")

    # Lazy import so optional extras are only needed when actually selected
    import importlib

    spec = STORAGE_BACKENDS[backend]
    module_path, class_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(url=url, ttl=ttl)


__all__ = ["Session", "SessionStore", "create_store"]
