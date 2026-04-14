"""Settings for deriva-mcp-ui.

All configuration is via DERIVA_CHATBOT_* environment variables.
"""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

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

    # Hostname remapping for server-side Credenza calls (token exchange, session, logout).
    # JSON object mapping public hostnames to internal Docker hostnames, e.g.
    # {"localhost":"deriva"}.  Mirrors DERIVA_MCP_HOSTNAME_MAP in deriva-mcp-core.
    hostname_map: dict[str, str] = Field(default_factory=dict)

    # Whether to verify TLS certificates on server-side Credenza calls.
    # Set to false for localhost where the dev CA cert does not cover the
    # internal Docker hostname.
    ssl_verify: bool = True

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

    # When True, unauthenticated users receive an anonymous session even when
    # Credenza is configured.  Authenticated login is still available via /login.
    # When False (default), Credenza presence forces login.
    allow_anonymous: bool = False

    # When True (default), users in LLM/local tier can toggle RAG-only mode
    # per session from the UI.  Set to False to lock the operating tier and
    # hide the toggle control entirely.
    allow_rag_toggle: bool = True

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

    def remap_url(self, url: str) -> str:
        """Rewrite url's hostname using hostname_map.

        Replaces the hostname component of url with the mapped value if a
        matching entry exists in hostname_map. Port is preserved. Useful for
        redirecting calls that use a public hostname (e.g. "localhost") to the
        corresponding internal network alias (e.g. "deriva") when running
        inside a Docker container where the public hostname resolves to the
        container itself.

        Returns url unchanged if no mapping applies.
        """
        if not self.hostname_map:
            return url
        parsed = urlparse(url)
        new_host = self.hostname_map.get(parsed.hostname or "", "")
        if not new_host:
            return url
        netloc = f"{new_host}:{parsed.port}" if parsed.port else new_host
        return urlunparse(parsed._replace(netloc=netloc))

    @property
    def credenza_configured(self) -> bool:
        """True when Credenza is configured and the OAuth login flow is available."""
        return bool(self.credenza_url)

    @property
    def auth_enabled(self) -> bool:
        """True when login is required -- Credenza configured and anonymous access not allowed."""
        return self.credenza_configured and not self.allow_anonymous

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

        if self.credenza_configured:
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
