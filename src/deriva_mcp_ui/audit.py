"""Structured JSON audit logging for deriva-mcp-ui.

Emits one JSON line per event to syslog (local1 facility) in Docker, or to
stderr as a fallback for local development.

Usage::

    from .audit import audit_event, init_audit_logger

    init_audit_logger(use_syslog=True)   # called once at startup

    audit_event("login_success", user_id="alice@example.org")
    audit_event("chat_request", user_id="alice@example.org", msg_len=42)
"""

from __future__ import annotations

import datetime
import logging
import os
from contextvars import ContextVar
from logging import StreamHandler
from logging.handlers import SysLogHandler
from typing import TYPE_CHECKING

from pythonjsonlogger import json as jsonlogger

if TYPE_CHECKING:
    from .storage.base import Session

_logger = logging.getLogger("deriva_mcp_ui.audit")
_initialized = False
_client_ip_var: ContextVar[str] = ContextVar("client_ip", default="unknown")


def init_audit_logger(use_syslog: bool = False) -> None:
    """Attach a JSON handler to the audit logger.

    Called once at application startup from server.main().  Subsequent calls
    are no-ops so tests can call it safely.

    Args:
        use_syslog: Route events to /dev/log (local1 facility) when True and
            the socket is available.  Falls back to stderr when unavailable.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    handler: logging.Handler = StreamHandler()  # fallback

    syslog_socket = "/dev/log"
    if use_syslog and os.path.exists(syslog_socket) and os.access(syslog_socket, os.W_OK):  # pragma: no cover
        try:
            handler = SysLogHandler(
                address=syslog_socket,
                facility=SysLogHandler.LOG_LOCAL1,
            )
            handler.ident = "deriva-mcp-ui-audit: "
        except Exception:
            handler = StreamHandler()

    formatter = jsonlogger.JsonFormatter("{message}", style="{", rename_fields={"message": "event"})
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def set_client_ip(ip: str) -> None:
    """Store the client IP for the current request in the contextvar."""
    _client_ip_var.set(ip)


def user_label(session: Session, include_email: bool = True) -> str:
    """Build a composite user identifier for LLM provider tracking and audit logs.

    Pass include_email=False when sending to LLM providers that reject email
    addresses in the user field (e.g. Anthropic).
    """
    cred = session.credenza_session or {}
    client_block = cred.get("client") or {}
    full_name = cred.get("full_name") or client_block.get("full_name") or ""
    email = cred.get("email") or client_block.get("email") or ""
    if include_email:
        if full_name or email:
            return f"{full_name} <{email}> ({session.user_id})"
    else:
        if full_name:
            return f"{full_name} ({session.user_id})"
    return session.user_id


def audit_event(event: str, **kwargs: object) -> None:
    """Emit a structured JSON audit event.

    Args:
        event: Event name (e.g. "login_success", "chat_request").
        **kwargs: Additional fields included in the log entry.
            Common fields: user_id, msg_len, duration_ms, error_type, reason.
    """
    entry = {
        "event": event,
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "client_ip": _client_ip_var.get(),
        **kwargs,
    }
    _logger.info(entry)
