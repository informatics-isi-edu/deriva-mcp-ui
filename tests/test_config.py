"""Tests for config.py."""

import pytest

from deriva_mcp_ui.config import Settings


def test_defaults():
    s = Settings()
    assert s.claude_model == "claude-haiku-4-5-20251001"
    assert s.max_history_turns == 10
    assert s.session_ttl == 28800
    assert s.storage_backend == "memory"
    assert s.debug is False


def test_default_catalog_mode_off():
    s = Settings()
    assert s.default_catalog_mode is False


def test_default_catalog_mode_on():
    s = Settings(default_hostname="example.org", default_catalog_id="1")
    assert s.default_catalog_mode is True


def test_validate_missing_required():
    s = Settings()
    with pytest.raises(ValueError, match="Missing required configuration"):
        s.validate_for_http()


def test_validate_partial_default_catalog():
    s = Settings(
        mcp_url="http://mcp",
        credenza_url="http://credenza",
        client_id="cid",
        mcp_resource="https://mcp.example.org",
        public_url="https://chatbot.example.org",
        anthropic_api_key="sk-ant-test",
        default_hostname="example.org",
        # default_catalog_id intentionally omitted
    )
    with pytest.raises(ValueError, match="must both be set"):
        s.validate_for_http()


def test_validate_passes():
    s = Settings(
        mcp_url="http://mcp",
        credenza_url="http://credenza",
        client_id="cid",
        mcp_resource="https://mcp.example.org",
        public_url="https://chatbot.example.org",
        anthropic_api_key="sk-ant-test",
    )
    s.validate_for_http()  # should not raise
