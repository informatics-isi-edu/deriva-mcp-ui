"""Settings for deriva-mcp-ui.

All configuration is via DERIVA_CHATBOT_* environment variables.
ANTHROPIC_API_KEY is the one exception -- it uses the standard env var name.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Anthropic API key -- standard env var name (no DERIVA_CHATBOT_ prefix)
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "DERIVA_CHATBOT_ANTHROPIC_API_KEY"),
    )

    # Default-catalog mode (both must be set to activate)
    default_hostname: str = ""
    default_catalog_id: str = ""
    default_catalog_label: str = ""

    # Tuning
    claude_model: str = "claude-haiku-4-5-20251001"
    max_history_turns: int = 10
    max_message_length: int = 10000
    session_ttl: int = 28800
    storage_backend: str = "memory"
    storage_backend_url: str = ""
    debug: bool = False

    @property
    def auth_enabled(self) -> bool:
        """True when Credenza is configured and the OAuth login flow is active."""
        return bool(self.credenza_url)

    def validate_for_http(self) -> None:
        """Raise ValueError if any required field is missing.

        When credenza_url is not set the server operates in anonymous mode:
        no login flow, no bearer token forwarded to the MCP server.  Only
        mcp_url and anthropic_api_key are required in that case.
        """
        if self.auth_enabled:
            required = {
                "DERIVA_CHATBOT_MCP_URL": self.mcp_url,
                "DERIVA_CHATBOT_CREDENZA_URL": self.credenza_url,
                "DERIVA_CHATBOT_CLIENT_ID": self.client_id,
                "DERIVA_CHATBOT_MCP_RESOURCE": self.mcp_resource,
                "DERIVA_CHATBOT_PUBLIC_URL": self.public_url,
                "ANTHROPIC_API_KEY": self.anthropic_api_key,
            }
        else:
            required = {
                "DERIVA_CHATBOT_MCP_URL": self.mcp_url,
                "ANTHROPIC_API_KEY": self.anthropic_api_key,
            }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

        only_one = bool(self.default_hostname) != bool(self.default_catalog_id)
        if only_one:
            raise ValueError(
                "DERIVA_CHATBOT_DEFAULT_HOSTNAME and DERIVA_CHATBOT_DEFAULT_CATALOG_ID"
                " must both be set (or both unset) to activate default-catalog mode"
            )

    @property
    def default_catalog_mode(self) -> bool:
        """True when both default-catalog vars are set."""
        return bool(self.default_hostname) and bool(self.default_catalog_id)
