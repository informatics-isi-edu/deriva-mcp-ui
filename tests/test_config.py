"""Tests for config.py."""

import pytest

from deriva_mcp_ui.config import Settings


def test_defaults():
    s = Settings()
    assert s.claude_model == "claude-haiku-4-5-latest"
    assert s.max_history_turns == 10
    assert s.session_ttl == 28800
    assert s.history_ttl == 604800
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


def test_branding_defaults():
    s = Settings()
    assert s.header_title == "DERIVA Chatbot"
    assert s.header_logo_url == "static/deriva-logo.png"


def test_branding_custom_title():
    s = Settings(header_title="FaceBase Portal")
    assert s.header_title == "FaceBase Portal"


def test_logo_url_https_valid():
    s = Settings(
        mcp_url="http://mcp",
        anthropic_api_key="sk-ant-test",
        header_logo_url="https://example.org/logo.png",
    )
    s.validate_for_http()  # should not raise


def test_logo_url_relative_static_valid():
    s = Settings(
        mcp_url="http://mcp",
        anthropic_api_key="sk-ant-test",
        header_logo_url="static/custom-logo.png",
    )
    s.validate_for_http()  # should not raise


def test_logo_url_rejects_relative_path_traversal():
    s = Settings(
        mcp_url="http://mcp",
        anthropic_api_key="sk-ant-test",
        header_logo_url="static/../../etc/passwd.png",
    )
    with pytest.raises(ValueError, match="must use HTTPS"):
        s.validate_for_http()


def test_logo_url_rejects_http():
    s = Settings(
        mcp_url="http://mcp",
        anthropic_api_key="sk-ant-test",
        header_logo_url="http://example.org/logo.png",
    )
    with pytest.raises(ValueError, match="must use HTTPS"):
        s.validate_for_http()


def test_logo_url_rejects_javascript():
    s = Settings(
        mcp_url="http://mcp",
        anthropic_api_key="sk-ant-test",
        header_logo_url="javascript:alert(1)",
    )
    with pytest.raises(ValueError, match="must use HTTPS"):
        s.validate_for_http()


def test_logo_url_rejects_non_image():
    s = Settings(
        mcp_url="http://mcp",
        anthropic_api_key="sk-ant-test",
        header_logo_url="https://example.org/script.js",
    )
    with pytest.raises(ValueError, match="must point to an image file"):
        s.validate_for_http()


def test_logo_url_accepts_all_image_types():
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico"):
        s = Settings(
            mcp_url="http://mcp",
            anthropic_api_key="sk-ant-test",
            header_logo_url=f"https://example.org/logo{ext}",
        )
        s.validate_for_http()  # should not raise
