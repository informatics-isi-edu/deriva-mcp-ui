"""Tests for chat.py: system prompt, history trimming, and the tool-calling loop."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deriva_mcp_ui.chat import (
    ChatCancelled,
    _extract_key_terms,
    _fetch_guides,
    _format_rag_response,
    _merge_rag_results,
    _prime_ermrest_syntax,
    _prime_schema,
    run_chat_turn,
    system_prompt,
    trim_history,
)
from deriva_mcp_ui.config import Settings
from deriva_mcp_ui.mcp_client import MCPAuthError
from deriva_mcp_ui.storage.base import Session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _mock_open_session(*_args, **_kwargs):
    """Yield a MagicMock that stands in for a ClientSession."""
    yield MagicMock()


def _settings(**kw) -> Settings:
    base = dict(
        mcp_url="http://mcp:8000",
        credenza_url="http://credenza",
        client_id="cid",
        mcp_resource="https://mcp.example.org",
        public_url="https://chatbot.example.org",
        llm_api_key="sk-test",
        llm_model="claude-sonnet-4-6",
        max_history_turns=5,
    )
    base.update(kw)
    return Settings(**base)


def _session(schema_primed: bool = True) -> Session:
    now = time.time()
    sess = Session(user_id="alice", bearer_token="tok", created_at=now, last_active=now)
    sess.schema_primed = schema_primed
    return sess


def _openai_tool(name: str) -> dict[str, Any]:
    """Return a tool definition in OpenAI function-calling format."""
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _collect_text(events: list[dict]) -> str:
    """Concatenate text content from run_chat_turn output."""
    return "".join(e["content"] for e in events if e.get("type") == "text")


def _collect_tool_events(events: list[dict]) -> list[dict]:
    """Extract tool_start/tool_end events from run_chat_turn output."""
    return [e for e in events if e.get("type") in ("tool_start", "tool_end")]


# ---------------------------------------------------------------------------
# LiteLLM streaming mock helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str | None = None,
) -> MagicMock:
    """Build a mock LiteLLM streaming chunk."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = delta
    chunk.choices[0].finish_reason = finish_reason
    return chunk


def _make_tool_call_delta(index: int, tc_id: str | None, name: str | None, arguments: str) -> MagicMock:
    """Build a mock tool call delta for streaming."""
    tc = MagicMock()
    tc.index = index
    tc.id = tc_id
    func = MagicMock()
    func.name = name
    func.arguments = arguments
    tc.function = func
    return tc


async def _async_iter(items):
    """Turn a list into an async iterator (simulates LiteLLM streaming)."""
    for item in items:
        yield item


def _text_stream(text_chunks: list[str], finish_reason: str = "stop"):
    """Build a stream of text chunks ending with the given finish_reason."""
    chunks = [_make_chunk(content=t) for t in text_chunks]
    if chunks:
        chunks[-1] = _make_chunk(content=text_chunks[-1], finish_reason=finish_reason)
    else:
        chunks.append(_make_chunk(finish_reason=finish_reason))
    return _async_iter(chunks)


def _tool_call_stream(tool_calls_data: list[dict], finish_reason: str = "tool_calls"):
    """Build a stream with tool call deltas.

    tool_calls_data: list of {"id": str, "name": str, "arguments": str}
    """
    chunks = []
    for i, tc in enumerate(tool_calls_data):
        # First chunk: id + name + start of arguments
        tc_delta = _make_tool_call_delta(i, tc["id"], tc["name"], tc["arguments"])
        chunks.append(_make_chunk(tool_calls=[tc_delta]))
    # Final chunk with finish_reason
    chunks.append(_make_chunk(finish_reason=finish_reason))
    return _async_iter(chunks)


# ---------------------------------------------------------------------------
# system_prompt
# ---------------------------------------------------------------------------


def test_system_prompt_general_mode():
    s = _settings()
    sess = _session()
    p = system_prompt(s, sess)
    assert "DERIVA data assistant" in p
    assert "hostname and catalog ID" in p


def test_system_prompt_default_catalog_mode():
    s = _settings(
        default_hostname="facebase.org", default_catalog_id="1", default_catalog_label="FaceBase"
    )
    sess = _session()
    p = system_prompt(s, sess)
    assert "FaceBase" in p
    assert "hostname and catalog ID" not in p


def test_system_prompt_default_catalog_uses_hostname_when_no_label():
    s = _settings(default_hostname="facebase.org", default_catalog_id="1")
    sess = _session()
    p = system_prompt(s, sess)
    assert "facebase.org" in p


# ---------------------------------------------------------------------------
# trim_history
# ---------------------------------------------------------------------------


def test_trim_history_under_limit():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert trim_history(msgs, max_turns=5) == msgs


def test_trim_history_at_limit():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert trim_history(msgs, max_turns=2) == msgs


def test_trim_history_over_limit():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    result = trim_history(msgs, max_turns=2)
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "q2"
    assert len([m for m in result if m["role"] == "user"]) == 2


def test_trim_history_starts_on_user():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    result = trim_history(msgs, max_turns=1)
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "q3"


def test_trim_history_preserves_tool_sequence():
    """A tool-calling sequence (assistant + tool messages) is kept intact."""
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "type": "function", "function": {"name": "get_schema", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "schema data"},
        {"role": "assistant", "content": "Here is the schema."},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    result = trim_history(msgs, max_turns=2)
    # Should keep from q2 onward (q2 is the second-to-last user message)
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "q2"
    assert len(result) == 6  # q2 + assistant(tool_calls) + tool + assistant + q3 + a3


def test_trim_history_empty():
    assert trim_history([], max_turns=5) == []


# ---------------------------------------------------------------------------
# run_chat_turn -- simple end_turn (no tools)
# ---------------------------------------------------------------------------


async def test_run_chat_turn_no_tools_yields_text():
    s = _settings()
    sess = _session()
    sess.tools = [_openai_tool("get_entities")]

    with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(
            return_value=_text_stream(["Hello ", "from LLM"])
        )
        mock_litellm.RateLimitError = litellm_rate_limit_error()
        mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()

        chunks = []
        async for chunk in run_chat_turn("hi", sess, s):
            chunks.append(chunk)

    assert _collect_text(chunks) == "Hello from LLM"
    # History updated: user + assistant
    assert len(sess.history) == 2
    assert sess.history[0] == {"role": "user", "content": "hi"}
    assert sess.history[1]["role"] == "assistant"


async def test_run_chat_turn_fetches_tools_when_none():
    s = _settings()
    sess = _session()
    assert sess.tools is None

    fake_tools = [_openai_tool("get_entities")]

    with patch("deriva_mcp_ui.chat.open_session", _mock_open_session):
        with patch("deriva_mcp_ui.chat.list_tools", AsyncMock(return_value=fake_tools)) as mock_lt:
            with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
                mock_litellm.acompletion = AsyncMock(
                    return_value=_text_stream(["ok"])
                )
                mock_litellm.RateLimitError = litellm_rate_limit_error()
                mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()

                async for _ in run_chat_turn("hi", sess, s):
                    pass

    mock_lt.assert_called_once()
    assert sess.tools == fake_tools


# ---------------------------------------------------------------------------
# run_chat_turn -- with tool call
# ---------------------------------------------------------------------------


async def test_run_chat_turn_with_tool_call():
    s = _settings()
    sess = _session()
    sess.tools = [_openai_tool("get_entities")]

    tc_args = json.dumps({"hostname": "h", "catalog_id": "1"})
    call_count = 0

    async def _acompletion_side_effect(**_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _tool_call_stream([{"id": "tid1", "name": "get_entities", "arguments": tc_args}])
        return _text_stream(["Done"])

    tool_result = "row1,row2"

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(return_value=tool_result)) as mock_ct:
        with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_acompletion_side_effect)
            mock_litellm.RateLimitError = litellm_rate_limit_error()
            mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()

            chunks = []
            async for chunk in run_chat_turn("show me data", sess, s):
                chunks.append(chunk)

    mock_ct.assert_called_once_with(
        "tok", "get_entities", {"hostname": "h", "catalog_id": "1"}, "http://mcp:8000", ssl_verify=True
    )
    assert "Done" in _collect_text(chunks)
    tool_evts = _collect_tool_events(chunks)
    assert any(e["type"] == "tool_start" and e["name"] == "get_entities" for e in tool_evts)
    assert any(e["type"] == "tool_end" and e["name"] == "get_entities" for e in tool_evts)
    # History: user, assistant (tool_calls), tool, assistant (end)
    assert len(sess.history) == 4


async def test_run_chat_turn_yields_tool_events():
    """tool_start and tool_end events are always emitted for every tool call."""
    s = _settings()
    sess = _session()
    sess.tools = [_openai_tool("get_entities")]

    tc_args = json.dumps({"limit": 5})
    call_count = 0

    async def _acompletion_side_effect(**_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _tool_call_stream([{"id": "tid-dbg", "name": "get_entities", "arguments": tc_args}])
        return _text_stream(["Done"])

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(return_value="row1,row2")):
        with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_acompletion_side_effect)
            mock_litellm.RateLimitError = litellm_rate_limit_error()
            mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()

            events = []
            async for event in run_chat_turn("hi", sess, s):
                events.append(event)

    tool_evts = _collect_tool_events(events)
    assert len(tool_evts) == 2
    start = next(e for e in tool_evts if e["type"] == "tool_start")
    end = next(e for e in tool_evts if e["type"] == "tool_end")
    assert start["name"] == "get_entities"
    assert start["input"] == {"limit": 5}
    assert end["name"] == "get_entities"
    assert "row1" in end["result"]


# ---------------------------------------------------------------------------
# run_chat_turn -- error cases
# ---------------------------------------------------------------------------


async def test_run_chat_turn_mcp_auth_error_propagates():
    s = _settings()
    sess = _session()
    sess.tools = [_openai_tool("tool")]

    tc_args = json.dumps({})
    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=MCPAuthError("401"))):
        with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                return_value=_tool_call_stream([{"id": "tid", "name": "tool", "arguments": tc_args}])
            )
            mock_litellm.RateLimitError = litellm_rate_limit_error()
            mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()

            with pytest.raises(MCPAuthError):
                async for _ in run_chat_turn("hi", sess, s):
                    pass


async def test_run_chat_turn_tool_error_continues():
    """A non-auth tool error should be returned as a tool_result string, not raised."""
    s = _settings()
    sess = _session()
    sess.tools = [_openai_tool("tool")]

    tc_args = json.dumps({})
    call_count = 0

    async def _acompletion_side_effect(**_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _tool_call_stream([{"id": "tid", "name": "tool", "arguments": tc_args}])
        return _text_stream(["Recovered"])

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=RuntimeError("oops"))):
        with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_acompletion_side_effect)
            mock_litellm.RateLimitError = litellm_rate_limit_error()
            mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()

            chunks = []
            async for chunk in run_chat_turn("hi", sess, s):
                chunks.append(chunk)

    assert "Recovered" in _collect_text(chunks)


# ---------------------------------------------------------------------------
# run_chat_turn -- history trimming
# ---------------------------------------------------------------------------


async def test_run_chat_turn_history_trimmed():
    s = _settings(max_history_turns=2)
    sess = _session()
    # Pre-load history with 5 turns (10 messages)
    sess.history = []
    for i in range(5):
        sess.history.append({"role": "user", "content": f"u{i}"})
        sess.history.append({"role": "assistant", "content": f"a{i}"})
    sess.tools = []

    with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(return_value=_text_stream(["hi"]))
        mock_litellm.RateLimitError = litellm_rate_limit_error()
        mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()

        async for _ in run_chat_turn("new", sess, s):
            pass

    # max_history_turns=2 -> at most 2 user messages in history
    user_msgs = [m for m in sess.history if m.get("role") == "user"]
    assert len(user_msgs) <= s.max_history_turns
    assert sess.history[0]["role"] == "user"


# ---------------------------------------------------------------------------
# Phase 5: schema priming
# ---------------------------------------------------------------------------


def _default_settings(**kw) -> Settings:
    base = dict(
        mcp_url="http://mcp:8000",
        credenza_url="http://credenza",
        client_id="cid",
        mcp_resource="https://mcp.example.org",
        public_url="https://chatbot.example.org",
        llm_api_key="sk-test",
        llm_model="claude-sonnet-4-6",
        max_history_turns=5,
        default_hostname="facebase.org",
        default_catalog_id="1",
    )
    base.update(kw)
    return Settings(**base)


async def test_prime_schema_fetches_all_schemas():
    """_prime_schema calls get_catalog_info then get_schema for each schema."""
    s = _default_settings()
    sess = _session()

    catalog_info = json.dumps({
        "hostname": "facebase.org", "catalog_id": "1",
        "schemas": [{"schema": "isa", "tables": 3, "comment": None}],
    })
    schema_detail = json.dumps({
        "schema": "isa", "comment": None,
        "tables": [{"table": "Study", "columns": [{"name": "RID"}]}],
    })

    call_log: list[str] = []

    async def _side_effect(token, name, args, url, **kw):
        call_log.append(name)
        if name == "get_catalog_info":
            return catalog_info
        if name == "get_schema":
            return schema_detail
        return ""

    with patch("deriva_mcp_ui.chat.call_tool", side_effect=_side_effect):
        result = await _prime_schema(sess, s)

    assert call_log == ["get_catalog_info", "get_schema"]
    assert "Study" in result


async def test_prime_schema_truncates_large_schemas():
    """Schema priming stops adding schemas once budget is exceeded."""
    from deriva_mcp_ui.chat import _SCHEMA_PRIMING_MAX_CHARS

    s = _default_settings()
    sess = _session()

    catalog_info = json.dumps({
        "hostname": "facebase.org", "catalog_id": "1",
        "schemas": [
            {"schema": "s1", "tables": 1, "comment": None},
            {"schema": "s2", "tables": 1, "comment": None},
        ],
    })
    # First schema fills the budget
    big_schema = "x" * (_SCHEMA_PRIMING_MAX_CHARS + 100)

    async def _side_effect(token, name, args, url, **kw):
        if name == "get_catalog_info":
            return catalog_info
        if name == "get_schema" and args.get("schema") == "s1":
            return big_schema
        return '{"schema": "s2", "tables": []}'

    with patch("deriva_mcp_ui.chat.call_tool", side_effect=_side_effect):
        result = await _prime_schema(sess, s)

    # Only s1 should be included (fills budget), s2 skipped
    assert result == big_schema


async def test_prime_schema_returns_empty_on_failure():
    s = _default_settings()
    sess = _session()

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=RuntimeError("down"))):
        result = await _prime_schema(sess, s)

    assert result == ""


def test_system_prompt_includes_schema_context():
    s = _default_settings()
    sess = _session()
    ctx = "Table: Dataset\nTable: Subject"
    p = system_prompt(s, sess, schema_context=ctx)
    assert "Available schema information" in p
    assert ctx in p


def test_system_prompt_no_schema_context_no_separator():
    s = _default_settings()
    sess = _session()
    p = system_prompt(s, sess, schema_context="")
    assert "Available schema information:" not in p
    # No standalone --- divider lines; the prompt may contain --- within rule
    # text as examples of what NOT to output, but not as actual section breaks.
    assert not re.search(r"(?m)^---$", p)


async def test_run_chat_turn_primes_schema_on_first_turn():
    s = _default_settings()
    sess = _session(schema_primed=False)
    sess.tools = []
    assert sess.schema_primed is False

    with patch("deriva_mcp_ui.chat.open_session", _mock_open_session), patch(
        "deriva_mcp_ui.chat._prime_schema", AsyncMock(return_value="Schema: isa")
    ) as mock_ps, patch(
        "deriva_mcp_ui.chat._fetch_guides", AsyncMock(return_value="")
    ), patch(
        "deriva_mcp_ui.chat._prime_ermrest_syntax", AsyncMock(return_value="")
    ):
        with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_text_stream(["ok"]))
            mock_litellm.RateLimitError = litellm_rate_limit_error()
            mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()
            async for _ in run_chat_turn("hello", sess, s):
                pass

    mock_ps.assert_called_once()
    assert sess.schema_primed is True


async def test_run_chat_turn_skips_priming_after_first_turn():
    s = _default_settings()
    sess = _session()
    sess.tools = []
    sess.schema_primed = True  # already primed

    with patch(
        "deriva_mcp_ui.chat._prime_schema", AsyncMock(return_value="Schema: isa")
    ) as mock_ps:
        with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_text_stream(["ok"]))
            mock_litellm.RateLimitError = litellm_rate_limit_error()
            mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()
            async for _ in run_chat_turn("hello again", sess, s):
                pass

    mock_ps.assert_not_called()


async def test_run_chat_turn_no_priming_in_general_mode():
    s = _settings()  # no default_hostname/catalog_id -> general mode
    sess = _session()
    sess.tools = []

    with patch("deriva_mcp_ui.chat._prime_schema", AsyncMock()) as mock_ps:
        with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_text_stream(["ok"]))
            mock_litellm.RateLimitError = litellm_rate_limit_error()
            mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()
            async for _ in run_chat_turn("hello", sess, s):
                pass

    mock_ps.assert_not_called()


# ---------------------------------------------------------------------------
# run_chat_turn -- retry on rate limit / service unavailable
# ---------------------------------------------------------------------------


def litellm_rate_limit_error():
    """Create a mock RateLimitError class."""
    return type("RateLimitError", (Exception,), {})


def litellm_service_unavailable_error():
    """Create a mock ServiceUnavailableError class."""
    return type("ServiceUnavailableError", (Exception,), {})


async def test_run_chat_turn_retries_on_service_unavailable():
    """A ServiceUnavailableError before any text is yielded triggers a retry."""
    s = _settings()
    sess = _session()
    sess.tools = [_openai_tool("tool")]

    call_count = 0
    _SUE = litellm_service_unavailable_error()

    async def _acompletion_side_effect(**_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _SUE("overloaded")
        return _text_stream(["Success after retry"])

    with patch("deriva_mcp_ui.chat.asyncio.sleep", AsyncMock()) as mock_sleep:
        with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_acompletion_side_effect)
            mock_litellm.RateLimitError = litellm_rate_limit_error()
            mock_litellm.ServiceUnavailableError = _SUE
            events = []
            async for event in run_chat_turn("hi", sess, s):
                events.append(event)

    assert call_count == 2
    mock_sleep.assert_called_once()
    assert "Success after retry" in _collect_text(events)


async def test_run_chat_turn_retries_on_rate_limit():
    """A RateLimitError before any text is yielded triggers a retry."""
    s = _settings()
    sess = _session()
    sess.tools = [_openai_tool("tool")]

    call_count = 0
    _RLE = litellm_rate_limit_error()

    async def _acompletion_side_effect(**_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _RLE("rate limited")
        return _text_stream(["Done"])

    with patch("deriva_mcp_ui.chat.asyncio.sleep", AsyncMock()):
        with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_acompletion_side_effect)
            mock_litellm.RateLimitError = _RLE
            mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()
            events = []
            async for event in run_chat_turn("hi", sess, s):
                events.append(event)

    assert call_count == 2
    assert "Done" in _collect_text(events)


async def test_run_chat_turn_raises_after_max_retries():
    """Exhausting all retries re-raises the last exception."""
    s = _settings()
    sess = _session()
    sess.tools = [_openai_tool("tool")]

    _SUE = litellm_service_unavailable_error()

    async def _always_overloaded(**_kw):
        raise _SUE("overloaded")

    with patch("deriva_mcp_ui.chat.asyncio.sleep", AsyncMock()):
        with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_always_overloaded)
            mock_litellm.RateLimitError = litellm_rate_limit_error()
            mock_litellm.ServiceUnavailableError = _SUE
            with pytest.raises(_SUE):
                async for _ in run_chat_turn("hi", sess, s):
                    pass


# ---------------------------------------------------------------------------
# run_chat_turn -- poll delay enforcement
# ---------------------------------------------------------------------------


async def test_run_chat_turn_poll_delay_on_repeated_tool():
    """When the same tool is called in two consecutive iterations, asyncio.sleep
    is called with _POLL_DELAY_SECONDS before executing the second iteration.
    First-call and single-use tools must not trigger the delay.
    """
    from deriva_mcp_ui.chat import _POLL_DELAY_SECONDS

    s = _settings()
    sess = _session()
    sess.tools = [_openai_tool("get_task_status")]

    tc_args = json.dumps({"task_id": "t1"})
    call_count = 0

    async def _acompletion_side_effect(**_kw):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return _tool_call_stream([{"id": f"tu_{call_count}", "name": "get_task_status", "arguments": tc_args}])
        return _text_stream(["Done"])

    sleep_mock = AsyncMock()
    with patch("deriva_mcp_ui.chat.asyncio.sleep", sleep_mock):
        with patch("deriva_mcp_ui.chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=_acompletion_side_effect)
            mock_litellm.RateLimitError = litellm_rate_limit_error()
            mock_litellm.ServiceUnavailableError = litellm_service_unavailable_error()
            with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(return_value="in progress")):
                events = []
                async for ev in run_chat_turn("check status", sess, s):
                    events.append(ev)

    # Sleep must have been called exactly once (between iteration 1 and 2, same tool)
    # and NOT called for the first iteration (new tool) or third (different stop_reason)
    poll_sleeps = [c for c in sleep_mock.call_args_list if c.args[0] == _POLL_DELAY_SECONDS]
    assert len(poll_sleeps) == 1


# ---------------------------------------------------------------------------
# Guide prompt fetching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_guides_returns_joined_prompts():
    s = _default_settings()
    sess = _session()

    async def _mock_get_prompt(token, name, url, **kw):
        return f"Guide for {name}"

    with patch("deriva_mcp_ui.chat.get_prompt", AsyncMock(side_effect=_mock_get_prompt)):
        result = await _fetch_guides(sess, s)

    assert "Guide for query_guide" in result
    assert "Guide for entity_guide" in result
    assert "Guide for annotation_guide" in result
    assert "Guide for catalog_guide" in result


@pytest.mark.asyncio
async def test_fetch_guides_skips_unavailable():
    s = _default_settings()
    sess = _session()

    async def _mock_get_prompt(token, name, url, **kw):
        if name == "query_guide":
            return "query text"
        return ""

    with patch("deriva_mcp_ui.chat.get_prompt", AsyncMock(side_effect=_mock_get_prompt)):
        result = await _fetch_guides(sess, s)

    assert result == "query text"


@pytest.mark.asyncio
async def test_fetch_guides_returns_empty_on_total_failure():
    s = _default_settings()
    sess = _session()

    with patch("deriva_mcp_ui.chat.get_prompt", AsyncMock(return_value="")):
        result = await _fetch_guides(sess, s)

    assert result == ""


# ---------------------------------------------------------------------------
# ERMrest syntax priming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prime_ermrest_syntax_returns_chunks():
    s = _default_settings()
    sess = _session()

    rag_result = json.dumps([
        {"source": "doc1.md", "text": "ERMrest filter syntax"},
        {"source": "doc2.md", "text": "ERMrest join syntax"},
    ])

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(return_value=rag_result)):
        result = await _prime_ermrest_syntax(sess, s)

    assert "ERMrest filter syntax" in result
    assert "ERMrest join syntax" in result


@pytest.mark.asyncio
async def test_prime_ermrest_syntax_deduplicates_by_source():
    s = _default_settings()
    sess = _session()

    # Both queries return the same source -- should appear only once
    rag_result = json.dumps([{"source": "same.md", "text": "same chunk"}])

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(return_value=rag_result)):
        result = await _prime_ermrest_syntax(sess, s)

    assert result.count("same chunk") == 1


@pytest.mark.asyncio
async def test_prime_ermrest_syntax_returns_empty_on_failure():
    s = _default_settings()
    sess = _session()

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=RuntimeError("down"))):
        result = await _prime_ermrest_syntax(sess, s)

    assert result == ""


@pytest.mark.asyncio
async def test_prime_ermrest_syntax_skips_error_results():
    s = _default_settings()
    sess = _session()

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(return_value="Error: rag not available")):
        result = await _prime_ermrest_syntax(sess, s)

    assert result == ""


# ---------------------------------------------------------------------------
# System prompt with guide and ermrest context
# ---------------------------------------------------------------------------


def test_system_prompt_includes_guide_context():
    s = _default_settings()
    sess = _session()
    p = system_prompt(s, sess, guide_context="QUERY GUIDE TEXT")
    assert "Tool usage guides:" in p
    assert "QUERY GUIDE TEXT" in p


def test_system_prompt_includes_ermrest_syntax():
    s = _default_settings()
    sess = _session()
    p = system_prompt(s, sess, ermrest_syntax="ERMrest path syntax")
    assert "ERMrest URL syntax reference:" in p
    assert "ERMrest path syntax" in p


def test_system_prompt_schema_loaded_rule_when_context_present():
    s = _default_settings()
    sess = _session()
    p = system_prompt(s, sess, schema_context="Table: Dataset")
    assert "SCHEMA IS ALREADY LOADED" in p


def test_system_prompt_schema_lookup_rule_when_no_context():
    s = _default_settings()
    sess = _session()
    p = system_prompt(s, sess, schema_context="")
    assert "SCHEMA LOOKUP" in p


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_chat_turn_stops_on_cancelled_event():
    """When the cancelled event is set, the loop raises ChatCancelled."""
    s = _settings()
    sess = _session()
    sess.tools = [_openai_tool("get_schema")]

    cancelled = asyncio.Event()
    cancelled.set()  # pre-set -- should stop immediately

    events = []
    with pytest.raises(ChatCancelled):
        with patch("deriva_mcp_ui.chat.litellm"):
            async for ev in run_chat_turn("hello", sess, s, cancelled=cancelled):
                events.append(ev)

    # No events should have been yielded -- cancelled before first LLM call
    assert len(events) == 0


# ---------------------------------------------------------------------------
# RAG-only mode
# ---------------------------------------------------------------------------


def _rag_settings(**kw) -> Settings:
    """Settings configured for RAG-only mode (no API key)."""
    base = dict(
        mcp_url="http://mcp:8000",
        mode="rag_only",
        max_history_turns=5,
        default_hostname="facebase.org",
        default_catalog_id="1",
    )
    base.update(kw)
    return Settings(**base)


def test_format_rag_response_with_results():
    results = [
        {"text": "ERMrest is a relational data service. It provides a RESTful HTTP interface to PostgreSQL.", "source": "docs/guide.md", "score": 0.85},
        {"text": "ERMrest supports entity retrieval and filtering via URL patterns.", "source": "docs/ref.md", "score": 0.70},
    ]
    resp = _format_rag_response("what is ERMrest?", results)
    assert "ERMrest is a relational data service" in resp
    # Per-source headings with relevance (grouped output uses #####)
    assert "#### **guide.md** (relevance: 0.85)" in resp
    assert "#### **ref.md** (relevance: 0.70)" in resp
    # Most relevant source appears first
    assert resp.index("guide.md") < resp.index("ref.md")


def test_format_rag_response_no_results():
    resp = _format_rag_response("unknown topic", [])
    assert "No relevant documentation found" in resp


def test_format_rag_response_deduplicates_sentences():
    """Identical sentences across sources appear only once (first source wins)."""
    results = [
        {"text": "ERMrest is a relational data service for scientific data.", "source": "a.md", "score": 0.9},
        {"text": "ERMrest is a relational data service for scientific data.", "source": "b.md", "score": 0.8},
    ]
    resp = _format_rag_response("what is ERMrest?", results)
    assert resp.count("ERMrest is a relational data service") == 1


def test_format_rag_response_with_schema():
    resp = _format_rag_response(
        "what tables exist?",
        [{"text": "The isa schema contains several important research tables.", "source": "doc.md", "score": 0.7}],
        schema_text="Schema: isa (3 tables)",
    )
    assert "Schema: isa" in resp
    assert "important research tables" in resp


def test_format_rag_response_schema_only():
    resp = _format_rag_response("show tables", [], schema_text="Schema: isa")
    assert "Schema: isa" in resp
    assert "No relevant documentation" not in resp


def test_format_rag_response_low_relevance_filtered_out():
    """Results below the minimum relevance threshold are excluded."""
    results = [
        {"text": "Some tangentially related content about data management.", "source": "a.md", "score": 0.15},
    ]
    resp = _format_rag_response("unrelated question", results)
    assert "No relevant documentation found" in resp


def test_format_rag_response_low_confidence_warning():
    """Results above threshold but below confidence level show a warning."""
    results = [
        {"text": "Some tangentially related content about data management.", "source": "a.md", "score": 0.35},
    ]
    resp = _format_rag_response("unrelated question", results)
    assert "not entirely certain" in resp


def test_format_rag_response_how_to_framing():
    results = [
        {"text": "First step is to navigate to the catalog page. Then select the schema you want.", "source": "a.md", "score": 0.8},
    ]
    resp = _format_rag_response("how do I browse schemas?", results)
    assert "process" in resp.lower() or "documentation says" in resp.lower()


def test_format_rag_response_data_framing():
    results = [
        {"text": "You can download datasets using the export feature in Chaise.", "source": "a.md", "score": 0.8},
    ]
    resp = _format_rag_response("how do I download data?", results)
    assert "download datasets" in resp


def test_format_rag_response_preserves_markdown_tables():
    """Chunks containing markdown tables are included verbatim, not sentence-split."""
    table_text = (
        "## Actions\n"
        "| Action | Path | Description |\n"
        "| --- | --- | --- |\n"
        "| read | /entity | Read entities |\n"
        "| create | /entity | Create entities |"
    )
    results = [{"text": table_text, "source": "docs/actions.md", "score": 0.8}]
    resp = _format_rag_response("what actions are available?", results)
    assert "| Action | Path | Description |" in resp
    assert "| read | /entity | Read entities |" in resp
    assert "#### **actions.md** (relevance: 0.80)" in resp


def test_format_rag_response_preserves_schema_listings():
    """Chunks with column type annotations are treated as structured."""
    listing = (
        "### Table: Dataset\n"
        "Columns:\n"
        "RID (ermrest_rid NOT NULL)\n"
        "Title (text NOT NULL)\n"
        "Description (text NOT NULL)\n"
        "Persistent_ID (text)"
    )
    results = [{"text": listing, "source": "schema:fb/1", "score": 0.9}]
    resp = _format_rag_response("describe the dataset table", results)
    assert "RID (ermrest_rid NOT NULL)" in resp
    assert "Title (text NOT NULL)" in resp


def test_format_rag_response_mixed_structured_and_prose():
    """When results contain both structured and prose chunks, both appear
    as separate source sections."""
    table_chunk = "| Column | Type |\n| --- | --- |\n| RID | text |\n| Title | text |"
    prose_chunk = "The Dataset table stores metadata about research datasets and their provenance."
    results = [
        {"text": table_chunk, "source": "schema.md", "score": 0.9},
        {"text": prose_chunk, "source": "guide.md", "score": 0.7},
    ]
    resp = _format_rag_response("what is the dataset table?", results)
    assert "| RID | text |" in resp
    assert "research datasets" in resp
    # Higher-scored source comes first
    assert resp.index("schema.md") < resp.index("guide.md")


def test_format_rag_response_source_ordering():
    """Sources are ordered by relevance score, most relevant first."""
    results = [
        {"text": "Lower relevance content about something tangential but still useful.", "source": "low.md", "score": 0.55},
        {"text": "High relevance content with the exact answer to the question.", "source": "high.md", "score": 0.90},
        {"text": "Medium relevance content providing some additional context here.", "source": "mid.md", "score": 0.65},
    ]
    resp = _format_rag_response("describe attributegroup", results)
    assert resp.index("high.md") < resp.index("mid.md")
    assert resp.index("mid.md") < resp.index("low.md")


def test_format_rag_response_threshold_hides_low_scores():
    """Sources below the relevance threshold are hidden with a note."""
    results = [
        {"text": "High relevance content with the exact answer.", "source": "high.md", "score": 0.85},
        {"text": "Below-threshold content that is only tangentially related here.", "source": "low.md", "score": 0.42},
    ]
    resp = _format_rag_response("what is ERMrest?", results)
    assert "#### **high.md** (relevance: 0.85)" in resp
    assert "#### **low.md**" not in resp
    assert "additional source" in resp
    assert "show all results" in resp


def test_format_rag_response_all_below_threshold_shown_anyway():
    """When every result is below the threshold, show them all rather than returning nothing."""
    results = [
        {"text": "Moderately related content about catalog exports.", "source": "a.md", "score": 0.42},
        {"text": "Somewhat related content about data models.", "source": "b.md", "score": 0.38},
    ]
    resp = _format_rag_response("what is ERMrest?", results)
    assert "#### **a.md** (relevance: 0.42)" in resp
    assert "#### **b.md** (relevance: 0.38)" in resp
    # No hidden-sources note since everything was shown
    assert "additional source" not in resp
    assert "show all results" not in resp


def test_format_rag_response_show_all_includes_low_scores():
    """show_all=True renders sources below the relevance threshold."""
    results = [
        {"text": "High relevance content with the exact answer.", "source": "high.md", "score": 0.85},
        {"text": "Below-threshold content that is only tangentially related here.", "source": "low.md", "score": 0.42},
    ]
    resp = _format_rag_response("what is ERMrest?", results, show_all=True)
    assert "#### **high.md** (relevance: 0.85)" in resp
    assert "#### **low.md** (relevance: 0.42)" in resp
    assert "additional source" not in resp


# ---------------------------------------------------------------------------
# Key term extraction and result merging
# ---------------------------------------------------------------------------


def test_extract_key_terms_strips_stop_words():
    result = _extract_key_terms("how does an ermrest attributegroup query work?")
    assert "ermrest" in result
    assert "attributegroup" in result
    assert "query" in result
    assert "how" not in result
    assert "does" not in result


def test_extract_key_terms_empty_for_short_question():
    assert _extract_key_terms("what is it?") == ""


def test_extract_key_terms_empty_when_mostly_content():
    # If almost all words are content words, skip secondary search
    assert _extract_key_terms("ermrest attributegroup") == ""


def test_merge_rag_results_deduplicates_by_source():
    primary = [
        {"text": "a", "source": "doc1.md", "score": 0.8},
        {"text": "b", "source": "doc2.md", "score": 0.6},
    ]
    secondary = [
        {"text": "c", "source": "doc2.md", "score": 0.7},  # dup source
        {"text": "d", "source": "doc3.md", "score": 0.9},
    ]
    merged = _merge_rag_results(primary, secondary)
    sources = [r["source"] for r in merged]
    assert sources.count("doc2.md") == 1
    assert "doc3.md" in sources


def test_merge_rag_results_sorted_by_score():
    primary = [{"text": "a", "source": "low.md", "score": 0.3}]
    secondary = [{"text": "b", "source": "high.md", "score": 0.9}]
    merged = _merge_rag_results(primary, secondary)
    assert merged[0]["source"] == "high.md"
    assert merged[1]["source"] == "low.md"


@pytest.mark.asyncio
async def test_rag_only_runs_secondary_search():
    """Multi-query: a secondary key-term search is run and results merged."""
    s = _rag_settings()
    sess = _session()

    call_log: list[tuple[str, dict]] = []

    async def _mock_call_tool(token, name, args, url, **kw):
        call_log.append((name, args))
        if name == "rag_search":
            return json.dumps([
                {"text": "Relevant documentation about the topic at hand.", "source": f"doc-{len(call_log)}.md", "score": 0.5},
            ])
        return ""

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=_mock_call_tool)):
        events = []
        async for ev in run_chat_turn("how does ermrest attributegroup query work?", sess, s):
            events.append(ev)

    # Should have four rag_search calls:
    # 1. primary (full question, no filter)
    # 2. secondary key-terms (no filter)
    # 3. supplemental web-content
    # 4. supplemental user-guide
    search_calls = [(n, a) for n, a in call_log if n == "rag_search"]
    assert len(search_calls) == 4
    unfiltered = [a for _, a in search_calls if "doc_type" not in a]
    assert len(unfiltered) == 2
    assert unfiltered[0]["query"] == "how does ermrest attributegroup query work?"
    assert "ermrest" in unfiltered[1]["query"]
    assert "attributegroup" in unfiltered[1]["query"]
    doc_types = {a["doc_type"] for _, a in search_calls if "doc_type" in a}
    assert doc_types == {"web-content", "user-guide"}


@pytest.mark.asyncio
async def test_run_chat_turn_rag_only_routing():
    """RAG-only mode calls _rag_only_response, not the LLM loop."""
    s = _rag_settings()
    sess = _session()
    assert s.operating_tier == "rag_only"

    rag_results = json.dumps([
        {"text": "You can query data using the entity browser in Chaise or the ERMrest API.", "source": "guide.md", "score": 0.8},
    ])

    async def _mock_call_tool(token, name, args, url, **kw):
        if name == "rag_search":
            return rag_results
        if name == "get_catalog_info":
            return "catalog info"
        return ""

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=_mock_call_tool)):
        events = []
        async for ev in run_chat_turn("how do I query data?", sess, s):
            events.append(ev)

    text = _collect_text(events)
    assert "query data using the entity browser" in text
    # History should have user + assistant messages
    assert len(sess.history) == 2
    assert sess.history[0]["role"] == "user"
    assert sess.history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_rag_only_override_on_session_routes_to_rag():
    """Session rag_only_override=True routes to RAG path even in LLM tier."""
    from deriva_mcp_ui.config import Settings as _S
    s = _S(mcp_url="http://mcp:8000", llm_api_key="sk-test",
           default_hostname="facebase.org", default_catalog_id="1",
           max_history_turns=5)
    assert s.operating_tier == "llm"

    sess = _session()
    sess.rag_only_override = True

    rag_results = json.dumps([
        {"text": "You can query data using the entity browser.", "source": "guide.md", "score": 0.8},
    ])

    async def _mock_call_tool(token, name, args, url, **kw):
        if name == "rag_search":
            return rag_results
        if name == "get_catalog_info":
            return "catalog info"
        return ""

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=_mock_call_tool)):
        events = []
        async for ev in run_chat_turn("how do I query data?", sess, s):
            events.append(ev)

    text = _collect_text(events)
    assert "entity browser" in text


@pytest.mark.asyncio
async def test_rag_only_schema_lookup_on_schema_question():
    """RAG-only mode also calls get_catalog_info when question mentions tables."""
    s = _rag_settings()
    sess = _session()

    call_log: list[str] = []

    async def _mock_call_tool(token, name, args, url, **kw):
        call_log.append(name)
        if name == "rag_search":
            return json.dumps([{"text": "The catalog contains research datasets organized by schema.", "source": "s", "score": 0.5}])
        if name == "get_catalog_info":
            return "Tables: Study, Dataset, Subject"
        return ""

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=_mock_call_tool)):
        events = []
        async for ev in run_chat_turn("what tables are available?", sess, s):
            events.append(ev)

    assert "get_catalog_info" in call_log
    assert "rag_search" in call_log
    text = _collect_text(events)
    assert "Tables: Study" in text


@pytest.mark.asyncio
async def test_rag_only_no_schema_for_non_schema_question():
    """RAG-only mode skips get_catalog_info for non-schema questions."""
    s = _rag_settings()
    sess = _session()

    call_log: list[str] = []

    async def _mock_call_tool(token, name, args, url, **kw):
        call_log.append(name)
        if name == "rag_search":
            return json.dumps([{"text": "answer", "source": "a", "score": 0.9}])
        return ""

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=_mock_call_tool)):
        events = []
        async for ev in run_chat_turn("how do I export data?", sess, s):
            events.append(ev)

    assert "get_catalog_info" not in call_log
    assert "rag_search" in call_log


@pytest.mark.asyncio
async def test_rag_only_handles_search_failure():
    """RAG-only mode yields fallback text when rag_search fails."""
    s = _rag_settings()
    sess = _session()

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=RuntimeError("down"))):
        events = []
        async for ev in run_chat_turn("anything", sess, s):
            events.append(ev)

    text = _collect_text(events)
    assert "No relevant documentation found" in text
