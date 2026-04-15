"""Tests for config.py."""

import pytest

from deriva_mcp_ui.config import Settings


def test_defaults():
    s = Settings()
    assert s.llm_model == "claude-haiku-4-5"
    assert s.max_history_turns == 10
    assert s.session_ttl == 28800
    assert s.history_ttl == 604800
    assert s.storage_backend == "memory"
    assert s.debug is False
    assert s.llm_provider == ""
    assert s.llm_api_base == ""
    assert s.hostname_map == {}
    assert s.ssl_verify is True


def test_remap_url_no_map():
    s = Settings()
    assert s.remap_url("https://localhost/authn") == "https://localhost/authn"


def test_remap_url_with_map():
    s = Settings(hostname_map={"localhost": "deriva"})
    assert s.remap_url("https://localhost/authn") == "https://deriva/authn"


def test_remap_url_preserves_port():
    s = Settings(hostname_map={"localhost": "deriva"})
    assert s.remap_url("https://localhost:8080/authn") == "https://deriva:8080/authn"


def test_remap_url_no_match():
    s = Settings(hostname_map={"localhost": "deriva"})
    assert s.remap_url("https://other.example.com/authn") == "https://other.example.com/authn"


def test_remap_url_mcp_url():
    s = Settings(hostname_map={"localhost": "deriva"})
    assert s.remap_url("https://localhost/mcp") == "https://deriva/mcp"


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
        llm_api_key="sk-test",
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
        llm_api_key="sk-test",
    )
    s.validate_for_http()  # should not raise


def test_branding_defaults():
    s = Settings()
    assert s.header_title == "DERIVA Data Assistant"
    assert s.header_logo_url == "static/deriva-logo.png"
    assert s.code_theme == "vs2015"


def test_branding_custom_title():
    s = Settings(header_title="FaceBase Portal")
    assert s.header_title == "FaceBase Portal"


def test_code_theme_override():
    s = Settings(code_theme="github-dark")
    assert s.code_theme == "github-dark"


def test_logo_url_https_valid():
    s = Settings(
        mcp_url="http://mcp",
        llm_api_key="sk-test",
        header_logo_url="https://example.org/logo.png",
    )
    s.validate_for_http()  # should not raise


def test_logo_url_relative_static_valid():
    s = Settings(
        mcp_url="http://mcp",
        llm_api_key="sk-test",
        header_logo_url="static/custom-logo.png",
    )
    s.validate_for_http()  # should not raise


def test_logo_url_rejects_relative_path_traversal():
    s = Settings(
        mcp_url="http://mcp",
        llm_api_key="sk-test",
        header_logo_url="static/../../etc/passwd.png",
    )
    with pytest.raises(ValueError, match="must use HTTPS"):
        s.validate_for_http()


def test_logo_url_rejects_http():
    s = Settings(
        mcp_url="http://mcp",
        llm_api_key="sk-test",
        header_logo_url="http://example.org/logo.png",
    )
    with pytest.raises(ValueError, match="must use HTTPS"):
        s.validate_for_http()


def test_logo_url_rejects_javascript():
    s = Settings(
        mcp_url="http://mcp",
        llm_api_key="sk-test",
        header_logo_url="javascript:alert(1)",
    )
    with pytest.raises(ValueError, match="must use HTTPS"):
        s.validate_for_http()


def test_logo_url_rejects_non_image():
    s = Settings(
        mcp_url="http://mcp",
        llm_api_key="sk-test",
        header_logo_url="https://example.org/script.js",
    )
    with pytest.raises(ValueError, match="must point to an image file"):
        s.validate_for_http()


def test_logo_url_accepts_all_image_types():
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico"):
        s = Settings(
            mcp_url="http://mcp",
            llm_api_key="sk-test",
            header_logo_url=f"https://example.org/logo{ext}",
        )
        s.validate_for_http()  # should not raise


# ---------------------------------------------------------------------------
# Operating tier detection
# ---------------------------------------------------------------------------


def test_operating_tier_auto_llm():
    s = Settings(llm_api_key="sk-test")
    assert s.operating_tier == "llm"


def test_operating_tier_auto_rag_only():
    s = Settings()
    assert s.operating_tier == "rag_only"


def test_operating_tier_auto_local():
    s = Settings(llm_provider="ollama", llm_model="llama3")
    assert s.operating_tier == "local"


def test_operating_tier_explicit_rag_only():
    s = Settings(llm_api_key="sk-test", mode="rag_only")
    assert s.operating_tier == "rag_only"


def test_operating_tier_explicit_llm():
    s = Settings(llm_api_key="sk-test", mode="llm")
    assert s.operating_tier == "llm"


def test_validate_rag_only_no_api_key_needed():
    """RAG-only mode only requires mcp_url, not an API key."""
    s = Settings(mcp_url="http://mcp", mode="rag_only")
    s.validate_for_http()  # should not raise


def test_validate_local_no_api_key_needed():
    """Ollama local mode does not require an API key."""
    s = Settings(
        mcp_url="http://mcp",
        llm_provider="ollama",
        llm_model="llama3",
    )
    s.validate_for_http()  # should not raise


def test_validate_llm_requires_api_key():
    s = Settings(mcp_url="http://mcp", mode="llm")
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        s.validate_for_http()
