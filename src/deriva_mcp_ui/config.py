"""Settings for deriva-mcp-ui.

All configuration is via DERIVA_CHATBOT_* environment variables.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOGO_ALLOWED_SCHEMES = ("https",)
_LOGO_ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DERIVA_CHATBOT_", case_sensitive=False, populate_by_name=True
    )

    # Required
    mcp_url: str = ""
    credenza_url: str = ""
    client_id: str = ""
    mcp_resource: str = ""
    public_url: str = ""

    # LLM provider configuration
    llm_api_key: str = Field(default="")
    llm_model: str = "claude-haiku-4-5"
    llm_provider: str = ""  # auto-detected from model string if empty
    llm_api_base: str = ""  # for Ollama, Azure, self-hosted endpoints

    # Operating mode: "auto" (detect from config), "rag_only", or "llm"
    mode: str = "auto"

    # Default-catalog mode (both must be set to activate)
    default_hostname: str = ""
    default_catalog_id: str = ""
    default_catalog_label: str = ""

    # Branding
    header_title: str = "DERIVA Data Assistant"
    header_logo_url: str = "static/deriva-logo.png"

    # Tuning
    max_history_turns: int = 10
    max_message_length: int = 10000
    session_ttl: int = 28800
    history_ttl: int = 604800  # 7 days
    storage_backend: str = "memory"
    storage_backend_url: str = ""
    debug: bool = False

    # App syslog: enable for non-Docker deployments where syslog is the
    # only path to a centralized collector.  Leave False under Docker
    # (compose driver: syslog already forwards stderr).
    app_use_syslog: bool = False

    # Access logging (uvicorn request log)
    access_logfile_path: str = "deriva-mcp-ui-access.log"
    access_use_syslog: bool = False

    @property
    def auth_enabled(self) -> bool:
        """True when Credenza is configured and the OAuth login flow is active."""
        return bool(self.credenza_url)

    @property
    def operating_tier(self) -> str:
        """Return the effective operating tier: 'llm', 'local', or 'rag_only'.

        When mode is 'auto', the tier is detected from configuration:
        - API key present -> 'llm' (cloud API)
        - Provider is 'ollama' with a model configured -> 'local'
        - Neither -> 'rag_only'

        Explicit mode='rag_only' or mode='llm' overrides auto-detection.
        """
        if self.mode == "rag_only":
            return "rag_only"
        if self.mode == "llm":
            return "llm"
        # auto-detection
        if self.llm_api_key:
            return "llm"
        if self.llm_provider == "ollama" and self.llm_model:
            return "local"
        return "rag_only"

    def validate_for_http(self) -> None:
        """Raise ValueError if any required field is missing.

        In RAG-only mode only mcp_url is required (no API key needed).
        In LLM mode an API key is required unless the provider is ollama.
        """
        # mcp_url is always required
        required: dict[str, str] = {"DERIVA_CHATBOT_MCP_URL": self.mcp_url}

        if self.auth_enabled:
            required.update({
                "DERIVA_CHATBOT_CREDENZA_URL": self.credenza_url,
                "DERIVA_CHATBOT_CLIENT_ID": self.client_id,
                "DERIVA_CHATBOT_MCP_RESOURCE": self.mcp_resource,
                "DERIVA_CHATBOT_PUBLIC_URL": self.public_url,
            })

        tier = self.operating_tier
        if tier in ("llm", "local") and tier != "local":
            required["DERIVA_CHATBOT_LLM_API_KEY"] = self.llm_api_key

        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

        only_one = bool(self.default_hostname) != bool(self.default_catalog_id)
        if only_one:
            raise ValueError(
                "DERIVA_CHATBOT_DEFAULT_HOSTNAME and DERIVA_CHATBOT_DEFAULT_CATALOG_ID"
                " must both be set (or both unset) to activate default-catalog mode"
            )

        if self.header_logo_url:
            is_relative = (
                self.header_logo_url.startswith("static/")
                and "/" not in self.header_logo_url[len("static/"):]
            )
            if not is_relative:
                parsed = urlparse(self.header_logo_url)
                if parsed.scheme not in _LOGO_ALLOWED_SCHEMES:
                    raise ValueError(
                        "DERIVA_CHATBOT_HEADER_LOGO_URL must use HTTPS or"
                        f" a static/ relative path (got scheme {parsed.scheme!r})"
                    )
            path_lower = self.header_logo_url.lower()
            if not any(path_lower.endswith(ext) for ext in _LOGO_ALLOWED_EXTENSIONS):
                raise ValueError(
                    "DERIVA_CHATBOT_HEADER_LOGO_URL must point to an image file"
                    f" ({', '.join(_LOGO_ALLOWED_EXTENSIONS)})"
                )

    @property
    def default_catalog_mode(self) -> bool:
        """True when both default-catalog vars are set."""
        return bool(self.default_hostname) and bool(self.default_catalog_id)
