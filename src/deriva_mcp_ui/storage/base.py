"""SessionStore protocol and Session dataclass."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Session:
    """Server-side session state for one authenticated user."""

    user_id: str
    bearer_token: str | None = None
    # Full Credenza /session response dict -- source of truth for display name,
    # email, groups, and any other identity fields the UI may need.
    credenza_session: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    input_history: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    created_at: float = 0.0
    last_active: float = 0.0
    # General-purpose mode: user-supplied catalog context
    gp_hostname: str = ""
    gp_catalog_id: str = ""
    # Schema priming: True once priming has been run for this conversation
    schema_primed: bool = False
    # Cached priming context -- injected into the system prompt on every turn
    primed_schema: str = ""
    primed_guides: str = ""
    primed_ermrest: str = ""
    # Per-session RAG-only override: when True, bypass LLM even if the server
    # is configured for LLM tier.
    rag_only_override: bool = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "user_id": self.user_id,
                "bearer_token": self.bearer_token,
                "credenza_session": self.credenza_session,
                "history": self.history,
                "input_history": self.input_history,
                "tools": self.tools,
                "created_at": self.created_at,
                "last_active": self.last_active,
                "gp_hostname": self.gp_hostname,
                "gp_catalog_id": self.gp_catalog_id,
                "schema_primed": self.schema_primed,
                "primed_schema": self.primed_schema,
                "primed_guides": self.primed_guides,
                "primed_ermrest": self.primed_ermrest,
                "rag_only_override": self.rag_only_override,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> Session:
        d = json.loads(data)
        return cls(
            user_id=d["user_id"],
            bearer_token=d["bearer_token"],
            credenza_session=d.get("credenza_session", {}),
            history=d.get("history", []),
            input_history=d.get("input_history", []),
            tools=d.get("tools"),
            created_at=d.get("created_at", 0.0),
            last_active=d.get("last_active", 0.0),
            gp_hostname=d.get("gp_hostname", ""),
            gp_catalog_id=d.get("gp_catalog_id", ""),
            schema_primed=d.get("schema_primed", False),
            primed_schema=d.get("primed_schema", ""),
            primed_guides=d.get("primed_guides", ""),
            primed_ermrest=d.get("primed_ermrest", ""),
            rag_only_override=d.get("rag_only_override", False),
        )


class SessionStore(Protocol):
    """Protocol for session store backends."""

    async def get(self, session_id: str) -> Session | None:
        """Return session if it exists and has not expired, else None."""
        ...  # pragma: no cover

    async def set(self, session_id: str, session: Session, ttl: int | None = None) -> None:
        """Persist or update a session.

        ttl overrides the store default when provided (seconds).
        """
        ...  # pragma: no cover

    async def delete(self, session_id: str) -> None:
        """Remove a session."""
        ...  # pragma: no cover

    async def sweep(self) -> None:
        """Evict expired sessions (called periodically by the memory backend)."""
        ...  # pragma: no cover
