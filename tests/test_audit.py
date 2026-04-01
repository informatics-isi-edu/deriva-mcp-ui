"""Tests for audit.py: init_audit_logger and audit_event."""

from __future__ import annotations

import logging

import pytest

from deriva_mcp_ui.audit import audit_event, init_audit_logger


@pytest.fixture(autouse=True)
def _reset_audit_logger():
    """Reset the audit logger state between tests."""
    import deriva_mcp_ui.audit as _audit
    original = _audit._initialized
    _audit._initialized = False
    # Clear any handlers added by a previous call
    logger = logging.getLogger("deriva_mcp_ui.audit")
    logger.handlers.clear()
    yield
    _audit._initialized = original


def test_init_audit_logger_adds_handler():
    logger = logging.getLogger("deriva_mcp_ui.audit")
    assert len(logger.handlers) == 0
    init_audit_logger(use_syslog=False)
    assert len(logger.handlers) == 1


def test_init_audit_logger_idempotent():
    """Calling init_audit_logger twice must not add a second handler."""
    init_audit_logger(use_syslog=False)
    init_audit_logger(use_syslog=False)
    logger = logging.getLogger("deriva_mcp_ui.audit")
    assert len(logger.handlers) == 1


def test_audit_event_emits_without_error():
    init_audit_logger(use_syslog=False)
    # Should not raise
    audit_event("test_event", user_id="alice", extra_field=42)


def test_audit_event_works_before_init():
    """audit_event must not raise even if init_audit_logger was not called."""
    audit_event("early_event", user_id="bob")
