"""Tests for mcp_client.py: tool listing, tool calls, and error handling."""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deriva_mcp_ui.mcp_client import (
    MCPAuthError,
    MCPConnectionError,
    _slim_parameters,
    call_tool,
    list_tools,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MCP_URL = "http://mcp-server:8000"
TOKEN = "bearer-test-token"


def _mock_tool(name: str, description: str = "", schema: dict | None = None) -> MagicMock:
    """Build a MagicMock that looks like an mcp.types.Tool."""
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema or {"type": "object", "properties": {}}
    return t


def _mock_text_block(text: str) -> MagicMock:
    """Build a MagicMock that looks like an mcp.types.TextContent."""
    from mcp.types import TextContent

    return TextContent(type="text", text=text)


def _mock_call_result(texts: list[str], is_error: bool = False) -> MagicMock:
    result = MagicMock()
    result.content = [_mock_text_block(t) for t in texts]
    result.isError = is_error
    return result


@contextlib.asynccontextmanager
async def _fake_connect(mock_session):
    """Context manager factory that yields a pre-built mock session."""
    yield mock_session


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------


async def test_list_tools_empty():
    session = AsyncMock()
    session.list_tools.return_value = MagicMock(tools=[])

    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_fake_connect(session)):
        result = await list_tools(TOKEN, MCP_URL)

    assert result == []


async def test_list_tools_converts_to_openai_format():
    """inputSchema (MCP) must become function.parameters (OpenAI format)."""
    schema = {"type": "object", "properties": {"hostname": {"type": "string"}}}
    session = AsyncMock()
    session.list_tools.return_value = MagicMock(
        tools=[_mock_tool("get_entities", "Get entities", schema)]
    )

    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_fake_connect(session)):
        result = await list_tools(TOKEN, MCP_URL)

    assert len(result) == 1
    tool = result[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_entities"
    assert tool["function"]["description"] == "Get entities"
    assert "inputSchema" not in tool
    # Property type preserved; no description/title injected on this fixture
    assert tool["function"]["parameters"]["properties"]["hostname"] == {"type": "string"}


async def test_list_tools_strips_property_descriptions():
    """Per-property description and title fields are stripped to reduce token cost."""
    schema = {
        "type": "object",
        "required": ["hostname"],
        "properties": {
            "hostname": {
                "type": "string",
                "description": "The DERIVA hostname",
                "title": "Hostname",
            },
            "catalog_id": {
                "type": "string",
                "description": "Catalog identifier",
            },
        },
    }
    session = AsyncMock()
    session.list_tools.return_value = MagicMock(
        tools=[_mock_tool("get_entities", "Get entities", schema)]
    )

    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_fake_connect(session)):
        result = await list_tools(TOKEN, MCP_URL)

    params = result[0]["function"]["parameters"]
    props = params["properties"]
    assert "description" not in props["hostname"]
    assert "title" not in props["hostname"]
    assert props["hostname"] == {"type": "string"}
    assert "description" not in props["catalog_id"]
    assert props["catalog_id"] == {"type": "string"}
    # Top-level schema fields are untouched
    assert params["type"] == "object"
    assert params["required"] == ["hostname"]


# ---------------------------------------------------------------------------
# _slim_parameters unit tests
# ---------------------------------------------------------------------------


def test_slim_schema_removes_description_and_title():
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "string", "description": "X value", "title": "X"},
            "y": {"type": "integer", "description": "Y value"},
        },
    }
    result = _slim_parameters(schema)
    assert result["properties"]["x"] == {"type": "string"}
    assert result["properties"]["y"] == {"type": "integer"}


def test_slim_schema_preserves_top_level_fields():
    schema = {
        "type": "object",
        "description": "Tool schema",
        "title": "MyTool",
        "required": ["x"],
        "properties": {"x": {"type": "string", "description": "ignored"}},
    }
    result = _slim_parameters(schema)
    assert result["type"] == "object"
    assert result["description"] == "Tool schema"
    assert result["title"] == "MyTool"
    assert result["required"] == ["x"]


def test_slim_schema_empty_properties():
    schema = {"type": "object", "properties": {}}
    result = _slim_parameters(schema)
    assert result == {"type": "object", "properties": {}}


def test_slim_schema_no_properties_key():
    schema = {"type": "object"}
    result = _slim_parameters(schema)
    assert result == {"type": "object"}


def test_slim_schema_empty_dict():
    assert _slim_parameters({}) == {}


def test_slim_schema_none():
    assert _slim_parameters(None) is None  # type: ignore[arg-type]


async def test_list_tools_multiple():
    session = AsyncMock()
    session.list_tools.return_value = MagicMock(
        tools=[
            _mock_tool("get_entities", "Get rows"),
            _mock_tool("get_schema", "Get schema"),
            _mock_tool("rag_search", "Search docs"),
        ]
    )

    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_fake_connect(session)):
        result = await list_tools(TOKEN, MCP_URL)

    assert [t["function"]["name"] for t in result] == ["get_entities", "get_schema", "rag_search"]


async def test_list_tools_no_description_omitted():
    """Tools with empty description should not carry a 'description' key in function."""
    session = AsyncMock()
    session.list_tools.return_value = MagicMock(tools=[_mock_tool("no_desc", description="")])

    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_fake_connect(session)):
        result = await list_tools(TOKEN, MCP_URL)

    assert "description" not in result[0]["function"]


async def test_list_tools_none_schema_becomes_empty_dict():
    """A tool with inputSchema=None should produce parameters={}."""
    t = MagicMock()
    t.name = "bare_tool"
    t.description = "desc"
    t.inputSchema = None

    session = AsyncMock()
    session.list_tools.return_value = MagicMock(tools=[t])

    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_fake_connect(session)):
        result = await list_tools(TOKEN, MCP_URL)

    assert result[0]["function"]["parameters"] == {}


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------


async def test_call_tool_single_text_block():
    session = AsyncMock()
    session.call_tool.return_value = _mock_call_result(["hello world"])

    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_fake_connect(session)):
        result = await call_tool(
            TOKEN, "get_entities", {"hostname": "h", "catalog_id": "1"}, MCP_URL
        )

    assert result == "hello world"
    session.call_tool.assert_called_once_with("get_entities", {"hostname": "h", "catalog_id": "1"})


async def test_call_tool_multiple_text_blocks_joined():
    session = AsyncMock()
    session.call_tool.return_value = _mock_call_result(["line one", "line two", "line three"])

    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_fake_connect(session)):
        result = await call_tool(TOKEN, "get_schema", {}, MCP_URL)

    assert result == "line one\nline two\nline three"


async def test_call_tool_empty_content():
    session = AsyncMock()
    session.call_tool.return_value = _mock_call_result([])

    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_fake_connect(session)):
        result = await call_tool(TOKEN, "noop", {}, MCP_URL)

    assert result == ""


async def test_call_tool_is_error_prefixes_message():
    session = AsyncMock()
    session.call_tool.return_value = _mock_call_result(["catalog not found"], is_error=True)

    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_fake_connect(session)):
        result = await call_tool(TOKEN, "get_entities", {}, MCP_URL)

    assert result.startswith("Error:")
    assert "catalog not found" in result


async def test_call_tool_non_text_block_placeholder():
    """Non-TextContent blocks should produce a bracketed type placeholder."""
    from mcp.types import TextContent

    non_text = MagicMock()
    non_text.__class__ = type("ImageContent", (), {})  # fake class name

    call_result = MagicMock()
    call_result.isError = False
    call_result.content = [TextContent(type="text", text="before"), non_text]

    session = AsyncMock()
    session.call_tool.return_value = call_result

    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_fake_connect(session)):
        result = await call_tool(TOKEN, "tool", {}, MCP_URL)

    assert "before" in result
    assert "[" in result  # placeholder bracket


# ---------------------------------------------------------------------------
# Error propagation through list_tools / call_tool
# ---------------------------------------------------------------------------
# _connect is patched to raise the already-converted typed exceptions.
# The httpx->typed conversion is tested separately in test__connect_* below.


@contextlib.asynccontextmanager
async def _raise(exc: Exception):
    raise exc
    yield  # pragma: no cover -- makes this an async generator


async def test_list_tools_raises_auth_error():
    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_raise(MCPAuthError("401"))):
        with pytest.raises(MCPAuthError):
            await list_tools(TOKEN, MCP_URL)


async def test_list_tools_raises_connection_error():
    with patch(
        "deriva_mcp_ui.mcp_client._connect", return_value=_raise(MCPConnectionError("refused"))
    ):
        with pytest.raises(MCPConnectionError):
            await list_tools(TOKEN, MCP_URL)


async def test_call_tool_raises_auth_error():
    with patch("deriva_mcp_ui.mcp_client._connect", return_value=_raise(MCPAuthError("401"))):
        with pytest.raises(MCPAuthError):
            await call_tool(TOKEN, "get_entities", {}, MCP_URL)


# ---------------------------------------------------------------------------
# _connect error conversion (httpx -> typed exceptions)
# ---------------------------------------------------------------------------


async def test_connect_converts_401_to_auth_error():
    import httpx

    from deriva_mcp_ui.mcp_client import _connect

    response = MagicMock()
    response.status_code = 401

    @contextlib.asynccontextmanager
    async def _fake_streamable(*_a, **_kw):
        raise httpx.HTTPStatusError("401", request=MagicMock(), response=response)
        yield  # pragma: no cover

    with patch("deriva_mcp_ui.mcp_client.streamablehttp_client", _fake_streamable):
        with pytest.raises(MCPAuthError):
            async with _connect(MCP_URL, TOKEN):
                pass


async def test_connect_converts_connect_error():
    import httpx

    from deriva_mcp_ui.mcp_client import _connect

    @contextlib.asynccontextmanager
    async def _fake_streamable(*_a, **_kw):
        raise httpx.ConnectError("refused")
        yield  # pragma: no cover

    with patch("deriva_mcp_ui.mcp_client.streamablehttp_client", _fake_streamable):
        with pytest.raises(MCPConnectionError):
            async with _connect(MCP_URL, TOKEN):
                pass


async def test_connect_converts_non_401_http_error():
    import httpx

    from deriva_mcp_ui.mcp_client import _connect

    response = MagicMock()
    response.status_code = 503

    @contextlib.asynccontextmanager
    async def _fake_streamable(*_a, **_kw):
        raise httpx.HTTPStatusError("503", request=MagicMock(), response=response)
        yield  # pragma: no cover

    with patch("deriva_mcp_ui.mcp_client.streamablehttp_client", _fake_streamable):
        with pytest.raises(MCPConnectionError):
            async with _connect(MCP_URL, TOKEN):
                pass
