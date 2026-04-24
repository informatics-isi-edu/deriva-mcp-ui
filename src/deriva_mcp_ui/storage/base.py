"""SessionStore protocol and Session dataclass."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Session:
    """Server-side session state for one authenticated user."""

    user_id: str
    # Stable identifier for this conversation session, used in audit logs.
    # Generated once at session creation and persisted.
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
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
    # Number of chat turns completed in this session.
    turn_count: int = 0
    # Approximate LLM spend and token counts accumulated during this session.
    # Reset when the session expires; use the store's lifetime totals for durable history.
    session_cost_usd: float = 0.0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_cache_read_tokens: int = 0
    session_cache_creation_tokens: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "user_id": self.user_id,
                "session_id": self.session_id,
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
                "turn_count": self.turn_count,
                "session_cost_usd": self.session_cost_usd,
                "session_prompt_tokens": self.session_prompt_tokens,
                "session_completion_tokens": self.session_completion_tokens,
                "session_cache_read_tokens": self.session_cache_read_tokens,
                "session_cache_creation_tokens": self.session_cache_creation_tokens,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> Session:
        d = json.loads(data)
        return cls(
            user_id=d["user_id"],
            session_id=d.get("session_id", str(uuid.uuid4())),
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
            turn_count=d.get("turn_count", 0),
            session_cost_usd=d.get("session_cost_usd", 0.0),
            session_prompt_tokens=d.get("session_prompt_tokens", 0),
            session_completion_tokens=d.get("session_completion_tokens", 0),
            session_cache_read_tokens=d.get("session_cache_read_tokens", 0),
            session_cache_creation_tokens=d.get("session_cache_creation_tokens", 0),
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

    async def increment_user_cost(
        self,
        user_id: str,
        cost_usd: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        """Atomically add cost and token counts to the durable lifetime totals for user_id."""
        ...  # pragma: no cover

    async def get_user_lifetime_cost(self, user_id: str) -> float:
        """Return the durable lifetime LLM spend total for user_id (0.0 if unknown)."""
        ...  # pragma: no cover

    async def upsert_user_identity(self, user_id: str, email: str, full_name: str) -> None:
        """Record or refresh identity info for user_id.

        Sets first_seen on insert; updates email, full_name, and last_seen on conflict.
        """
        ...  # pragma: no cover

    async def get_user_last_seen(self, user_id: str) -> float | None:
        """Return the last_seen Unix timestamp for user_id, or None if not recorded."""
        ...  # pragma: no cover
