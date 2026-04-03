"""Tests for chat.py: system prompt, history trimming, and the tool-calling loop."""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deriva_mcp_ui.chat import (
    _fetch_guides,
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
        anthropic_api_key="sk-ant-test",
        claude_model="claude-sonnet-4-6",
        max_history_turns=5,
    )
    base.update(kw)
    return Settings(**base)


def _session(schema_primed: bool = True) -> Session:
    now = time.time()
    sess = Session(user_id="alice", bearer_token="tok", created_at=now, last_active=now)
    sess.schema_primed = schema_primed
    return sess


def _tool_block(tool_id: str, name: str, inp: dict) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = inp
    return b


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.model_dump.return_value = {"type": "text", "text": text}
    return b


def _final_message(content: list, stop_reason: str = "end_turn") -> MagicMock:
    msg = MagicMock()
    msg.stop_reason = stop_reason
    msg.content = content
    return msg


async def _text_chunks(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


def _collect_text(events: list[dict]) -> str:
    """Concatenate text content from run_chat_turn output."""
    return "".join(e["content"] for e in events if e.get("type") == "text")


def _collect_tool_events(events: list[dict]) -> list[dict]:
    """Extract tool_start/tool_end events from run_chat_turn output."""
    return [e for e in events if e.get("type") in ("tool_start", "tool_end")]


@contextlib.asynccontextmanager
async def _mock_stream(text_chunks: list[str], final_message: Any):
    """Fake anthropic streaming context manager."""
    stream = MagicMock()
    stream.text_stream = _text_chunks(text_chunks)
    stream.get_final_message = AsyncMock(return_value=final_message)
    yield stream


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
    msgs = [{"role": "user"}, {"role": "assistant"}] * 3
    assert trim_history(msgs, max_turns=5) == msgs


def test_trim_history_at_limit():
    msgs = [{"role": "user"}, {"role": "assistant"}] * 5
    assert trim_history(msgs, max_turns=5) == msgs


def test_trim_history_over_limit():
    msgs = [{"role": "user", "content": str(i)} for i in range(12)]
    # Alternate user/assistant
    for i, m in enumerate(msgs):
        m["role"] = "user" if i % 2 == 0 else "assistant"
    result = trim_history(msgs, max_turns=4)
    assert len(result) <= 8
    assert result[0]["role"] == "user"


def test_trim_history_starts_on_user():
    # Even if slicing would land on an assistant message, result must start with user
    msgs = []
    for i in range(10):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    result = trim_history(msgs, max_turns=3)
    assert result[0]["role"] == "user"


def test_trim_history_skips_orphaned_tool_result():
    """Slicing must not leave a tool_result block whose tool_use was trimmed away.

    Actual conversation pattern:
      user_A -> assistant_with_tool_use -> tool_result_user -> assistant_after_tools
             -> plain_user -> assistant_reply

    With max_turns=2 (limit=4), slicing gives:
      [tool_result_user, assistant_after_tools, plain_user, assistant_reply]

    tail[0] is tool_result_user whose tool_use was trimmed -- must skip it and
    the following assistant reply, landing on [plain_user, assistant_reply].
    """
    tool_result_user = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_123", "content": "ok"}],
    }
    assistant_after_tools = {"role": "assistant", "content": [{"type": "text", "text": "done"}]}
    plain_user = {"role": "user", "content": "next question"}
    assistant_reply = {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}

    msgs = [
        {"role": "user", "content": "first question"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_123", "name": "get_schema", "input": {}}],
        },
        tool_result_user,  # orphaned if trim starts here
        assistant_after_tools,
        plain_user,
        assistant_reply,
    ]
    # max_turns=2 -> limit=4 -> tail = [tool_result_user, assistant_after_tools, plain_user, assistant_reply]
    # while loop skips tool_result_user + assistant_after_tools (tail[2:]) -> [plain_user, assistant_reply]
    result = trim_history(msgs, max_turns=2)
    assert result == [plain_user, assistant_reply]
    assert result[0]["role"] == "user"
    # no tool_result blocks in result
    for msg in result:
        content = msg.get("content")
        if isinstance(content, list):
            assert not any(b.get("type") == "tool_result" for b in content)


def test_trim_history_orphaned_tool_result_at_exact_limit():
    """When the tail begins on an assistant message (odd slice), advance past it too."""
    # Simulate: slice starts on an assistant message, not a tool_result user msg.
    plain_user = {"role": "user", "content": "hello"}
    assistant_reply = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}

    msgs = [
        {"role": "assistant", "content": [{"type": "text", "text": "stale"}]},
        plain_user,
        assistant_reply,
    ]
    # limit=2: tail = [assistant_stale, plain_user] -- starts on assistant, skip it
    result = trim_history(msgs, max_turns=1)
    assert result[0]["role"] == "user"
    assert result[0] == plain_user


# ---------------------------------------------------------------------------
# run_chat_turn -- simple end_turn (no tools)
# ---------------------------------------------------------------------------


async def test_run_chat_turn_no_tools_yields_text():
    s = _settings()
    sess = _session()
    sess.tools = [{"name": "get_entities", "input_schema": {}}]

    final = _final_message([_text_block("Hello from Claude")], stop_reason="end_turn")

    with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.stream.return_value = _mock_stream(["Hello ", "from Claude"], final)

        chunks = []
        async for chunk in run_chat_turn("hi", sess, s):
            chunks.append(chunk)

    assert _collect_text(chunks) == "Hello from Claude"
    # History updated
    assert len(sess.history) == 2
    assert sess.history[0] == {"role": "user", "content": "hi"}
    assert sess.history[1]["role"] == "assistant"


async def test_run_chat_turn_fetches_tools_when_none():
    s = _settings()
    sess = _session()
    assert sess.tools is None

    final = _final_message([_text_block("ok")], stop_reason="end_turn")
    fake_tools = [{"name": "get_entities", "input_schema": {}}]

    with patch("deriva_mcp_ui.chat.open_session", _mock_open_session):
        with patch("deriva_mcp_ui.chat.list_tools", AsyncMock(return_value=fake_tools)) as mock_lt:
            with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.messages.stream.return_value = _mock_stream(["ok"], final)

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
    sess.tools = [{"name": "get_entities", "input_schema": {}}]

    tool_block = _tool_block("tid1", "get_entities", {"hostname": "h", "catalog_id": "1"})
    tool_block.model_dump.return_value = {
        "type": "tool_use",
        "id": "tid1",
        "name": "get_entities",
        "input": {"hostname": "h", "catalog_id": "1"},
    }

    # First response: tool_use; second response: end_turn
    final_tool = _final_message([tool_block], stop_reason="tool_use")
    final_end = _final_message([_text_block("Done")], stop_reason="end_turn")

    call_count = 0

    @contextlib.asynccontextmanager
    async def _stream_factory(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            async with _mock_stream([], final_tool) as s_:
                yield s_
        else:
            async with _mock_stream(["Done"], final_end) as s_:
                yield s_

    tool_result = "row1,row2"

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(return_value=tool_result)) as mock_ct:
        with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream = _stream_factory

            chunks = []
            async for chunk in run_chat_turn("show me data", sess, s):
                chunks.append(chunk)

    mock_ct.assert_called_once_with(
        "tok", "get_entities", {"hostname": "h", "catalog_id": "1"}, "http://mcp:8000"
    )
    assert "Done" in _collect_text(chunks)
    tool_evts = _collect_tool_events(chunks)
    assert any(e["type"] == "tool_start" and e["name"] == "get_entities" for e in tool_evts)
    assert any(e["type"] == "tool_end" and e["name"] == "get_entities" for e in tool_evts)
    # History should include user, assistant (tool_use), user (tool_result), assistant (end)
    assert len(sess.history) == 4


async def test_run_chat_turn_yields_tool_events():
    """tool_start and tool_end events are always emitted for every tool call."""
    s = _settings()
    sess = _session()
    sess.tools = [{"name": "get_entities", "input_schema": {}}]

    tool_block = _tool_block("tid-dbg", "get_entities", {"limit": 5})
    tool_block.model_dump.return_value = {
        "type": "tool_use",
        "id": "tid-dbg",
        "name": "get_entities",
        "input": {"limit": 5},
    }
    final_tool = _final_message([tool_block], stop_reason="tool_use")
    final_end = _final_message([_text_block("Done")], stop_reason="end_turn")

    call_count = 0

    @contextlib.asynccontextmanager
    async def _stream_factory(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            async with _mock_stream([], final_tool) as s_:
                yield s_
        else:
            async with _mock_stream(["Done"], final_end) as s_:
                yield s_

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(return_value="row1,row2")):
        with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value.messages.stream = _stream_factory
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
    sess.tools = [{"name": "tool", "input_schema": {}}]

    tool_block = _tool_block("tid", "tool", {})
    tool_block.model_dump.return_value = {
        "type": "tool_use",
        "id": "tid",
        "name": "tool",
        "input": {},
    }
    final_tool = _final_message([tool_block], stop_reason="tool_use")

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=MCPAuthError("401"))):
        with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = _mock_stream([], final_tool)

            with pytest.raises(MCPAuthError):
                async for _ in run_chat_turn("hi", sess, s):
                    pass


async def test_run_chat_turn_tool_error_continues():
    """A non-auth tool error should be returned as a tool_result string, not raised."""
    s = _settings()
    sess = _session()
    sess.tools = [{"name": "tool", "input_schema": {}}]

    tool_block = _tool_block("tid", "tool", {})
    tool_block.model_dump.return_value = {
        "type": "tool_use",
        "id": "tid",
        "name": "tool",
        "input": {},
    }
    final_tool = _final_message([tool_block], stop_reason="tool_use")
    final_end = _final_message([_text_block("Recovered")], stop_reason="end_turn")

    call_count = 0

    @contextlib.asynccontextmanager
    async def _stream_factory(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            async with _mock_stream([], final_tool) as s_:
                yield s_
        else:
            async with _mock_stream(["Recovered"], final_end) as s_:
                yield s_

    with patch("deriva_mcp_ui.chat.call_tool", AsyncMock(side_effect=RuntimeError("oops"))):
        with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream = _stream_factory

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
    # Pre-load history with 10 messages (5 turns)
    sess.history = []
    for i in range(5):
        sess.history.append({"role": "user", "content": f"u{i}"})
        sess.history.append({"role": "assistant", "content": f"a{i}"})
    sess.tools = []

    final = _final_message([_text_block("hi")], stop_reason="end_turn")

    with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.stream.return_value = _mock_stream(["hi"], final)

        async for _ in run_chat_turn("new", sess, s):
            pass

    # max_history_turns=2 means at most 4 messages (2 pairs) + the new turn = 6 messages -> trimmed to 4
    assert len(sess.history) <= (s.max_history_turns * 2) + 2
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
        anthropic_api_key="sk-ant-test",
        claude_model="claude-sonnet-4-6",
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
    import json

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
    import json

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
    assert "---" not in p


async def test_run_chat_turn_primes_schema_on_first_turn():
    s = _default_settings()
    sess = _session(schema_primed=False)
    sess.tools = []
    assert sess.schema_primed is False

    final = _final_message([_text_block("ok")], stop_reason="end_turn")

    with patch("deriva_mcp_ui.chat.open_session", _mock_open_session), patch(
        "deriva_mcp_ui.chat._prime_schema", AsyncMock(return_value="Schema: isa")
    ) as mock_ps, patch(
        "deriva_mcp_ui.chat._fetch_guides", AsyncMock(return_value="")
    ), patch(
        "deriva_mcp_ui.chat._prime_ermrest_syntax", AsyncMock(return_value="")
    ):
        with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = _mock_stream(["ok"], final)
            async for _ in run_chat_turn("hello", sess, s):
                pass

    mock_ps.assert_called_once()
    assert sess.schema_primed is True


async def test_run_chat_turn_skips_priming_after_first_turn():
    s = _default_settings()
    sess = _session()
    sess.tools = []
    sess.schema_primed = True  # already primed

    final = _final_message([_text_block("ok")], stop_reason="end_turn")

    with patch(
        "deriva_mcp_ui.chat._prime_schema", AsyncMock(return_value="Schema: isa")
    ) as mock_ps:
        with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = _mock_stream(["ok"], final)
            async for _ in run_chat_turn("hello again", sess, s):
                pass

    mock_ps.assert_not_called()


async def test_run_chat_turn_no_priming_in_general_mode():
    s = _settings()  # no default_hostname/catalog_id -> general mode
    sess = _session()
    sess.tools = []

    final = _final_message([_text_block("ok")], stop_reason="end_turn")

    with patch("deriva_mcp_ui.chat._prime_schema", AsyncMock()) as mock_ps:
        with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = _mock_stream(["ok"], final)
            async for _ in run_chat_turn("hello", sess, s):
                pass

    mock_ps.assert_not_called()


# ---------------------------------------------------------------------------
# run_chat_turn -- retry on 429 / 529
# ---------------------------------------------------------------------------


async def test_run_chat_turn_retries_on_overloaded():
    """A 529 overloaded error before any text is yielded triggers a retry."""
    import anthropic as _anthropic

    s = _settings()
    sess = _session()
    sess.tools = [{"name": "tool", "input_schema": {}}]

    final = _final_message([_text_block("Success after retry")], stop_reason="end_turn")

    call_count = 0

    @contextlib.asynccontextmanager
    async def _stream_factory(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _anthropic.APIStatusError(
                "overloaded",
                response=MagicMock(status_code=529, headers={}),
                body={"type": "error", "error": {"type": "overloaded_error"}},
            )
        async with _mock_stream(["Success after retry"], final) as s_:
            yield s_

    with patch("deriva_mcp_ui.chat.asyncio.sleep", AsyncMock()) as mock_sleep:
        with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value.messages.stream = _stream_factory
            events = []
            async for event in run_chat_turn("hi", sess, s):
                events.append(event)

    assert call_count == 2
    mock_sleep.assert_called_once()
    assert "Success after retry" in _collect_text(events)


async def test_run_chat_turn_retries_on_rate_limit():
    """A 429 rate-limit error before any text is yielded triggers a retry."""
    import anthropic as _anthropic

    s = _settings()
    sess = _session()
    sess.tools = [{"name": "tool", "input_schema": {}}]

    final = _final_message([_text_block("Done")], stop_reason="end_turn")

    call_count = 0

    @contextlib.asynccontextmanager
    async def _stream_factory(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _anthropic.RateLimitError(
                "rate limited",
                response=MagicMock(status_code=429, headers={}),
                body={},
            )
        async with _mock_stream(["Done"], final) as s_:
            yield s_

    with patch("deriva_mcp_ui.chat.asyncio.sleep", AsyncMock()):
        with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value.messages.stream = _stream_factory
            events = []
            async for event in run_chat_turn("hi", sess, s):
                events.append(event)

    assert call_count == 2
    assert "Done" in _collect_text(events)


async def test_run_chat_turn_raises_after_max_retries():
    """Exhausting all retries re-raises the last exception."""
    import anthropic as _anthropic

    s = _settings()
    sess = _session()
    sess.tools = [{"name": "tool", "input_schema": {}}]

    @contextlib.asynccontextmanager
    async def _always_overloaded(*_a, **_kw):
        raise _anthropic.APIStatusError(
            "overloaded",
            response=MagicMock(status_code=529, headers={}),
            body={},
        )
        yield  # pragma: no cover

    with patch("deriva_mcp_ui.chat.asyncio.sleep", AsyncMock()):
        with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value.messages.stream = _always_overloaded
            with pytest.raises(_anthropic.APIStatusError):
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
    sess.tools = [{"name": "get_task_status", "input_schema": {}}]

    # Turn 1: Claude calls get_task_status -> "in progress"
    # Turn 2: Claude calls get_task_status again -> "complete", then end_turn
    tool_use_block_1 = MagicMock()
    tool_use_block_1.type = "tool_use"
    tool_use_block_1.id = "tu_1"
    tool_use_block_1.name = "get_task_status"
    tool_use_block_1.input = {"task_id": "t1"}

    tool_use_block_2 = MagicMock()
    tool_use_block_2.type = "tool_use"
    tool_use_block_2.id = "tu_2"
    tool_use_block_2.name = "get_task_status"
    tool_use_block_2.input = {"task_id": "t1"}

    first_tool_response = _final_message([tool_use_block_1], stop_reason="tool_use")
    second_tool_response = _final_message([tool_use_block_2], stop_reason="tool_use")
    end_response = _final_message([_text_block("Done")], stop_reason="end_turn")

    call_count = 0

    @contextlib.asynccontextmanager
    async def _rotating_stream(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            final = first_tool_response
            chunks: list[str] = []
        elif call_count == 2:
            final = second_tool_response
            chunks = []
        else:
            final = end_response
            chunks = ["Done"]
        stream = MagicMock()
        stream.text_stream = _text_chunks(chunks)
        stream.get_final_message = AsyncMock(return_value=final)
        yield stream

    sleep_mock = AsyncMock()
    with patch("deriva_mcp_ui.chat.asyncio.sleep", sleep_mock):
        with patch("deriva_mcp_ui.chat.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value.messages.stream = _rotating_stream
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
    import json

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
    import json

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
