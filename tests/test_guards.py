"""Tests for session guards: max_tool_rounds, send_timeout, tool_call_timeout, max_turn_tokens."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from toolstream._agent import AgentDefinition
from toolstream._direct import DirectClient
from toolstream._invoke import _build_invocation_config
from toolstream._tools import Tool, tool
from toolstream.config import SessionConfig
from toolstream.events import Error, StepFinish, StepStart, Text, ToolUse

from .conftest import direct_config, text_response, tool_call_response


# ============================================================
# Helpers
# ============================================================


def _noop_tool() -> Tool:
    """A tool that returns immediately."""

    async def noop(x: int = 0) -> str:
        return "ok"

    return Tool(
        name="noop",
        description="Does nothing",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        },
        handler=noop,
        inject=[],
    )


def _slow_tool(sleep_seconds: float = 10.0) -> Tool:
    """A tool that sleeps for a long time."""

    async def slow(x: int = 0) -> str:
        await asyncio.sleep(sleep_seconds)
        return "done"

    return Tool(
        name="slow",
        description="Sleeps a long time",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        },
        handler=slow,
        inject=[],
    )


def _repeating_mock(*responses: dict) -> httpx.AsyncClient:
    """Mock HTTP client that cycles through responses, repeating the last one forever."""
    responses_list = list(responses)

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        idx = min(call_count, len(responses_list) - 1)
        call_count += 1
        return httpx.Response(200, json=responses_list[idx])

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ============================================================
# test_max_tool_rounds_stops_loop
# ============================================================


@pytest.mark.asyncio
async def test_max_tool_rounds_stops_loop(tmp_path):
    """When max_tool_rounds=3, the loop stops after 3 iterations with an Error."""
    # LLM always returns a tool call -- without the guard, this would loop forever
    mock_client = _repeating_mock(
        tool_call_response("noop", {"x": 1}, call_id="call_1"),
    )
    config = direct_config(cwd=str(tmp_path), max_tool_rounds=3)
    client = DirectClient(config, tools=[_noop_tool()], http_client=mock_client)

    events = []
    async for event in client.send("do stuff"):
        events.append(event)

    # Should have tool use events for the rounds that ran
    tool_events = [e for e in events if isinstance(e, ToolUse)]
    step_finishes = [e for e in events if isinstance(e, StepFinish)]
    error_events = [e for e in events if isinstance(e, Error)]

    # 3 rounds means 3 API calls + tool dispatches, then on iteration 4 the guard fires
    assert len(tool_events) == 3
    assert len(step_finishes) == 3
    assert len(error_events) == 1
    assert error_events[0].name == "max_iterations_exceeded"
    assert "3" in error_events[0].message


# ============================================================
# test_send_timeout_stops_loop
# ============================================================


@pytest.mark.asyncio
async def test_send_timeout_stops_loop(tmp_path):
    """When send_timeout is exceeded, the loop stops with a timeout Error."""

    async def slow_noop(x: int = 0) -> str:
        await asyncio.sleep(0.15)
        return "ok"

    slow_noop_tool = Tool(
        name="slow_noop",
        description="Sleeps briefly",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        },
        handler=slow_noop,
        inject=[],
    )

    mock_client = _repeating_mock(
        tool_call_response("slow_noop", {"x": 1}, call_id="call_1"),
    )
    config = direct_config(cwd=str(tmp_path), send_timeout=0.5)
    client = DirectClient(config, tools=[slow_noop_tool], http_client=mock_client)

    events = []
    async for event in client.send("do stuff"):
        events.append(event)

    error_events = [e for e in events if isinstance(e, Error)]
    assert len(error_events) == 1
    assert error_events[0].name == "send_timeout_exceeded"
    assert "0.5" in error_events[0].message


# ============================================================
# test_tool_call_timeout_catches_hung_tool
# ============================================================


@pytest.mark.asyncio
async def test_tool_call_timeout_catches_hung_tool(tmp_path):
    """A tool that hangs is interrupted by tool_call_timeout, and the loop continues."""
    # First call triggers the slow tool, LLM sees the error and responds with text
    mock_client = _repeating_mock(
        tool_call_response("slow", {"x": 1}, call_id="call_1"),
        text_response("ok I give up"),
    )
    config = direct_config(cwd=str(tmp_path), tool_call_timeout=0.1)
    client = DirectClient(config, tools=[_slow_tool(10.0)], http_client=mock_client)

    events = []
    async for event in client.send("do stuff"):
        events.append(event)

    # The tool use event should have the timeout error in output
    tool_events = [e for e in events if isinstance(e, ToolUse)]
    assert len(tool_events) == 1
    assert "timed out" in tool_events[0].output
    assert "0.1" in tool_events[0].output

    # The loop continued: we got text after the timeout
    text_events = [e for e in events if isinstance(e, Text)]
    assert len(text_events) == 1
    assert text_events[0].text == "ok I give up"


# ============================================================
# test_tool_timeout_on_tool_overrides_session
# ============================================================


@pytest.mark.asyncio
async def test_tool_timeout_on_tool_overrides_session(tmp_path):
    """Per-tool timeout=0.1 takes precedence over session tool_call_timeout=10.0."""

    async def slow_handler(x: int = 0) -> str:
        await asyncio.sleep(10.0)
        return "done"

    tool_with_timeout = Tool(
        name="slow_with_timeout",
        description="Slow tool with per-tool timeout",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        },
        handler=slow_handler,
        inject=[],
        timeout=0.1,
    )

    mock_client = _repeating_mock(
        tool_call_response("slow_with_timeout", {"x": 1}, call_id="call_1"),
        text_response("done"),
    )
    # Session has generous timeout, but the tool's own timeout is tight
    config = direct_config(cwd=str(tmp_path), tool_call_timeout=10.0)
    client = DirectClient(config, tools=[tool_with_timeout], http_client=mock_client)

    events = []
    async for event in client.send("do stuff"):
        events.append(event)

    tool_events = [e for e in events if isinstance(e, ToolUse)]
    assert len(tool_events) == 1
    assert "timed out" in tool_events[0].output
    assert "0.1" in tool_events[0].output


# ============================================================
# test_build_invocation_config_forwards_guards
# ============================================================


def test_build_invocation_config_forwards_guards():
    """All 4 guard fields are forwarded to the child config."""
    config = SessionConfig(
        model="test-model",
        api_key="test-key",
        base_url="https://test.example.com",
        system_prompt="test",
        auth_style="x-api-key",
        max_tool_rounds=5,
        send_timeout=30.0,
        tool_call_timeout=10.0,
        max_turn_tokens=50000,
    )
    definition = AgentDefinition(
        name="test-agent",
        prompt_template="You are a test agent.",
        version="1.0",
        tools=None,
        model=None,
    )
    child = _build_invocation_config(definition, config)

    assert child.max_tool_rounds == 5
    assert child.send_timeout == 30.0
    assert child.tool_call_timeout == 10.0
    assert child.max_turn_tokens == 50000


# ============================================================
# test_no_guards_by_default
# ============================================================


@pytest.mark.asyncio
async def test_no_guards_by_default(tmp_path, mock_llm_responses):
    """Default SessionConfig has all guards as None; a normal send() works without errors."""
    config = direct_config(cwd=str(tmp_path))
    assert config.max_tool_rounds is None
    assert config.send_timeout is None
    assert config.tool_call_timeout is None
    assert config.max_turn_tokens is None

    # One tool call then a text response -- normal flow
    mock_client = mock_llm_responses(
        tool_call_response("noop", {"x": 1}, call_id="call_1"),
        text_response("all done"),
    )
    client = DirectClient(config, tools=[_noop_tool()], http_client=mock_client)

    events = []
    async for event in client.send("do stuff"):
        events.append(event)

    error_events = [e for e in events if isinstance(e, Error)]
    assert len(error_events) == 0

    # Normal flow completed
    text_events = [e for e in events if isinstance(e, Text)]
    assert len(text_events) == 1
    assert text_events[0].text == "all done"

    step_finishes = [e for e in events if isinstance(e, StepFinish)]
    assert len(step_finishes) == 2  # one for tool call round, one for final


# ============================================================
# test_max_turn_tokens_stops_loop
# ============================================================


@pytest.mark.asyncio
async def test_max_turn_tokens_stops_loop(tmp_path):
    """When cumulative tokens exceed max_turn_tokens, the loop stops."""
    # Each response uses 10+5=15 tokens. Budget of 20 means the first round
    # (15 tokens) passes, but the second round (30 cumulative) exceeds the budget.
    mock_client = _repeating_mock(
        tool_call_response("noop", {"x": 1}, call_id="call_1", prompt_tokens=10, completion_tokens=5),
    )
    config = direct_config(cwd=str(tmp_path), max_turn_tokens=20)
    client = DirectClient(config, tools=[_noop_tool()], http_client=mock_client)

    events = []
    async for event in client.send("do stuff"):
        events.append(event)

    error_events = [e for e in events if isinstance(e, Error)]
    assert len(error_events) == 1
    assert error_events[0].name == "token_budget_exceeded"
    assert "30" in error_events[0].message  # cumulative: 15 + 15 = 30
    assert "20" in error_events[0].message  # budget

    # First round completed normally, second triggered the guard
    step_finishes = [e for e in events if isinstance(e, StepFinish)]
    assert len(step_finishes) == 2
