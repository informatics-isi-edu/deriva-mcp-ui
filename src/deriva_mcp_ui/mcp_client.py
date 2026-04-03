"""MCP client wrapper.

Provides public coroutines used by the chat layer:

    list_tools(bearer_token, mcp_url) -> list[AnthropicTool]
    call_tool(bearer_token, name, arguments, mcp_url) -> str
    get_prompt(bearer_token, name, mcp_url) -> str
    open_session(bearer_token, mcp_url) -> async context manager

Each standalone call opens a fresh streamablehttp_client connection and
discards it when done, matching the server's stateless_http=True model
(no persistent session to maintain or reconnect).

For batching multiple calls on a single connection, use open_session() and
pass the yielded session to call_tool / get_prompt / list_tools via the
optional ``session`` parameter.

Typed exceptions
----------------
MCPConnectionError  -- server unreachable or transport-level failure
MCPAuthError        -- server returned 401 (bearer token rejected)
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent

logger = logging.getLogger(__name__)

# Type alias matching what the Anthropic SDK expects for tool definitions
AnthropicTool = dict[str, Any]


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------


class MCPConnectionError(Exception):
    """MCP server is unreachable or returned a non-auth transport error."""


class MCPAuthError(Exception):
    """MCP server rejected the bearer token (HTTP 401)."""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _connect(mcp_url: str, bearer_token: str | None):
    """Open a ClientSession to the MCP server and yield it, initialised.

    Translates transport-level errors into MCPConnectionError / MCPAuthError
    so callers do not need to handle httpx internals.

    When bearer_token is None the Authorization header is omitted, which is
    correct for servers running in allow-anonymous mode.
    """
    headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
    try:
        async with streamablehttp_client(
            mcp_url,
            headers=headers,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise MCPAuthError(f"MCP server rejected token: {exc}") from exc
        raise MCPConnectionError(f"MCP server error {exc.response.status_code}: {exc}") from exc
    except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
        raise MCPConnectionError(f"Cannot reach MCP server at {mcp_url}: {exc}") from exc


@contextlib.asynccontextmanager
async def open_session(bearer_token: str | None, mcp_url: str):
    """Open a reusable MCP session for batching multiple calls.

    Usage::

        async with open_session(token, url) as sess:
            tools = await list_tools(token, url, session=sess)
            result = await call_tool(token, "my_tool", {}, url, session=sess)
    """
    async with _connect(mcp_url, bearer_token) as session:
        yield session


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _slim_input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip per-property description and title fields from an input_schema.

    The Anthropic API accepts but does not require these fields.  Removing
    them from every property definition cuts the fixed tool-list token cost
    significantly (typically 30-50%) without affecting Claude's ability to
    call the tools correctly.

    Top-level schema fields (type, required, description, title) are kept
    intact -- only per-property fields inside 'properties' are trimmed.
    """
    if not schema or not isinstance(schema, dict):
        return schema
    props = schema.get("properties")
    if not isinstance(props, dict):
        return schema
    slimmed: dict[str, Any] = {}
    for prop_name, prop_def in props.items():
        if isinstance(prop_def, dict):
            slimmed[prop_name] = {k: v for k, v in prop_def.items() if k not in ("description", "title")}
        else:
            slimmed[prop_name] = prop_def
    return {**schema, "properties": slimmed}


async def list_tools(
    bearer_token: str | None,
    mcp_url: str,
    *,
    session: ClientSession | None = None,
) -> list[AnthropicTool]:
    """Return the MCP server's tool list in Anthropic schema format.

    MCP uses camelCase 'inputSchema'; Anthropic expects 'input_schema'.
    Per-property description and title fields are stripped from each tool's
    input_schema to reduce the fixed token cost of including the tool list
    in every API call.

    Pass an existing ``session`` (from open_session) to avoid opening a new
    connection.
    """
    if session is not None:
        result = await session.list_tools()
    else:
        async with _connect(mcp_url, bearer_token) as sess:
            result = await sess.list_tools()

    tools: list[AnthropicTool] = []
    for tool in result.tools:
        entry: AnthropicTool = {
            "name": tool.name,
            "input_schema": _slim_input_schema(tool.inputSchema if tool.inputSchema is not None else {})
        }
        if tool.description:
            entry["description"] = tool.description
        tools.append(entry)

    logger.debug("list_tools: %d tools from %s", len(tools), mcp_url)
    return tools


async def get_prompt(
    bearer_token: str | None,
    name: str,
    mcp_url: str,
    *,
    session: ClientSession | None = None,
) -> str:
    """Fetch an MCP prompt by name and return its text content.

    Returns the concatenated text of all messages in the prompt.
    Returns an empty string if the prompt is not found or has no content.

    Pass an existing ``session`` (from open_session) to avoid opening a new
    connection.
    """
    try:
        if session is not None:
            result = await session.get_prompt(name)
        else:
            async with _connect(mcp_url, bearer_token) as sess:
                result = await sess.get_prompt(name)
        parts: list[str] = []
        for msg in result.messages:
            if hasattr(msg.content, "text"):
                parts.append(msg.content.text)
            elif isinstance(msg.content, str):
                parts.append(msg.content)
        text = "\n".join(parts)
        logger.debug("get_prompt %s: %d chars", name, len(text))
        return text
    except Exception as exc:
        logger.debug("get_prompt %s failed: %s", name, exc)
        return ""


async def call_tool(
    bearer_token: str | None,
    name: str,
    arguments: dict[str, Any],
    mcp_url: str,
    *,
    session: ClientSession | None = None,
) -> str:
    """Invoke a tool on the MCP server and return its output as a string.

    All TextContent blocks in the result are concatenated.  Non-text blocks
    (images, embedded resources) are represented as placeholder strings so
    the chat layer always receives a plain string.

    If the tool result carries isError=True the returned string is prefixed
    with 'Error: ' so Claude can report it clearly.

    Pass an existing ``session`` (from open_session) to avoid opening a new
    connection.
    """
    if session is not None:
        result = await session.call_tool(name, arguments)
    else:
        async with _connect(mcp_url, bearer_token) as sess:
            result = await sess.call_tool(name, arguments)

    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            # ImageContent, EmbeddedResource, etc. -- not renderable in chat
            parts.append(f"[{type(block).__name__}]")

    text = "\n".join(parts) if parts else ""

    if result.isError:
        logger.warning("call_tool %s returned isError=True: %s", name, text[:200])
        return f"Error: {text}"

    logger.debug("call_tool %s: %d chars returned", name, len(text))
    return text
