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
from logging import StreamHandler
from logging.handlers import SysLogHandler

from pythonjsonlogger import json as jsonlogger

_logger = logging.getLogger("deriva_mcp_ui.audit")
_initialized = False


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

    formatter = jsonlogger.JsonFormatter(
        "{message}", style="{", rename_fields={"message": "event"}
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


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
        **kwargs,
    }
    _logger.info(entry)