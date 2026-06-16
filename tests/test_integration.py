"""Integration tests exercising the full toolstream stack with mock LLM responder."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from toolstream._agent import AgentDefinition, ToolRef
from toolstream._direct import DirectClient
from toolstream._invoke import invoke_agent
from toolstream._session import AsyncSession, SyncSession
from toolstream._tools import Tool, tool
from toolstream.events import Result, StepFinish, StepStart, Text, ToolUse

from .conftest import direct_config, text_response, tool_call_response


async def test_text_only_turn(mock_llm_responses):
    mock_client = mock_llm_responses(text_response("Hello there"))
    config = direct_config()

    client = DirectClient(config, http_client=mock_client, tools=[])
    events = []
    async for event in client.send("hi"):
        events.append(event)

    assert len(events) == 3
    assert isinstance(events[0], StepStart)
    assert isinstance(events[1], Text)
    assert events[1].text == "Hello there"
    assert isinstance(events[2], StepFinish)
    assert events[2].reason == "stop"
    assert events[2].input_tokens == 10
    assert events[2].output_tokens == 5


async def test_tool_call_then_text(mock_llm_responses):
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    greet_tool = Tool(
        name="greet",
        description="Greet someone",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        handler=greet,
        inject=[],
    )

    mock_client = mock_llm_responses(
        tool_call_response("greet", {"name": "Alice"}, call_id="call_greet"),
        text_response("I greeted Alice for you."),
    )
    config = direct_config()

    client = DirectClient(config, http_client=mock_client, tools=[greet_tool])
    events = []
    async for event in client.send("greet Alice"):
        events.append(event)

    # Expected: StepStart, ToolUse, StepFinish(tool-calls), Text, StepFinish(stop)
    assert isinstance(events[0], StepStart)

    assert isinstance(events[1], ToolUse)
    assert events[1].tool == "greet"
    assert events[1].call_id == "call_greet"
    assert events[1].output == "Hello, Alice!"
    assert events[1].input == {"name": "Alice"}

    assert isinstance(events[2], StepFinish)
    assert events[2].reason == "tool-calls"

    assert isinstance(events[3], Text)
    assert events[3].text == "I greeted Alice for you."

    assert isinstance(events[4], StepFinish)
    assert events[4].reason == "stop"


async def test_context_injection_end_to_end(mock_llm_responses):
    @dataclass
    class Ctx:
        db_conn: str

    received_values: dict = {}

    async def save_item(db_conn: str, item: str) -> str:
        received_values["db_conn"] = db_conn
        received_values["item"] = item
        return f"Saved {item}"

    save_tool = Tool(
        name="save_item",
        description="Save an item",
        input_schema={
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
        },
        handler=save_item,
        inject=["db_conn"],
    )

    mock_client = mock_llm_responses(
        tool_call_response("save_item", {"item": "widget"}),
        text_response("Done"),
    )
    config = direct_config()

    client = DirectClient(
        config,
        http_client=mock_client,
        tools=[save_tool],
        tool_context=Ctx(db_conn="postgres://test"),
    )
    events = []
    async for event in client.send("save widget"):
        events.append(event)

    assert received_values["db_conn"] == "postgres://test"
    assert received_values["item"] == "widget"

    tool_use_events = [e for e in events if isinstance(e, ToolUse)]
    assert len(tool_use_events) == 1
    assert tool_use_events[0].output == "Saved widget"


async def test_tool_error_handling(mock_llm_responses):
    async def failing_tool(x: int) -> str:
        raise ValueError("something went wrong")

    fail_tool = Tool(
        name="failing_tool",
        description="A tool that fails",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
        handler=failing_tool,
        inject=[],
    )

    mock_client = mock_llm_responses(
        tool_call_response("failing_tool", {"x": 42}),
        text_response("I encountered an error"),
    )
    config = direct_config()

    client = DirectClient(config, http_client=mock_client, tools=[fail_tool])
    events = []
    async for event in client.send("do something"):
        events.append(event)

    tool_use_events = [e for e in events if isinstance(e, ToolUse)]
    assert len(tool_use_events) == 1
    assert "Error: something went wrong" in tool_use_events[0].output


async def test_invoke_agent_setup():
    @tool()
    async def lookup(query: str) -> str:
        """Look up information."""
        return f"Result for {query}"

    definition = AgentDefinition(
        name="test-agent",
        prompt_template="You are a {role} assistant.",
        version="1.0",
        tools=[ToolRef("lookup")],
    )
    config = direct_config()
    available_tools = {"lookup": lookup}

    async with invoke_agent(
        definition,
        config,
        variables={"role": "helpful"},
        available_tools=available_tools,
    ) as session:
        assert session._config.system_prompt == "You are a helpful assistant."
        assert session._config.model == "test-model"
        assert session._config.tools is not None
        assert len(session._config.tools) == 1
        assert session._config.tools[0].name == "lookup"
        assert session._config.tools[0] is lookup._tool


async def test_token_accumulation_across_steps(mock_llm_responses):
    """Token counts from multiple StepFinish events in one turn must accumulate."""
    async def noop(x: str) -> str:
        return "ok"

    noop_tool = Tool(
        name="noop",
        description="No-op tool",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        handler=noop,
        inject=[],
    )

    # Two-step response: tool call (100+50 tokens), then text (200+80 tokens)
    mock_client = mock_llm_responses(
        tool_call_response("noop", {"x": "a"}, prompt_tokens=100, completion_tokens=50),
        text_response("done", prompt_tokens=200, completion_tokens=80),
    )

    config = direct_config(tools=[noop_tool])

    # Build AsyncSession and inject the mock http_client into its DirectClient
    session = AsyncSession(config)
    session._direct = DirectClient(
        config, http_client=mock_client, tools=[noop_tool],
    )

    events = []
    async for event in session.send("go"):
        events.append(event)

    result_events = [e for e in events if isinstance(e, Result)]
    assert len(result_events) == 1
    result = result_events[0]

    # DirectClient emits running totals in each StepFinish (not per-step
    # deltas). The first StepFinish has 100 input / 50 output, the second
    # has 300 input / 130 output (100+200, 50+80). AsyncSession._send_direct
    # accumulates these via +=, so the Result reflects the sum of all
    # StepFinish values. Before the bug fix, = (assignment) meant only the
    # last StepFinish's tokens were counted (300/130 instead of 400/180).
    assert result.total_input_tokens == 400  # 100 + (100+200)
    assert result.total_output_tokens == 180  # 50 + (50+80)

    await session.close()


def test_sync_session_streaming(mock_llm_responses):
    """SyncSession.send() yields events incrementally (queue-based streaming)."""
    mock_client = mock_llm_responses(text_response("hello back"))
    config = direct_config()

    with SyncSession(config) as session:
        # Inject mock http_client into the inner AsyncSession's DirectClient
        session._async_session._direct = DirectClient(
            config, http_client=mock_client, tools=[],
        )

        events = list(session.send("hi"))

    event_types = [type(e) for e in events]
    assert StepStart in event_types
    assert Text in event_types
    assert StepFinish in event_types
    assert Result in event_types

    text_events = [e for e in events if isinstance(e, Text)]
    assert text_events[0].text == "hello back"


def test_sync_session_error_propagation(mock_llm_responses):
    """Errors from the async producer propagate to the sync caller."""

    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal server error"})

    error_client = httpx.AsyncClient(transport=httpx.MockTransport(error_handler))
    config = direct_config()

    with SyncSession(config) as session:
        session._async_session._direct = DirectClient(
            config, http_client=error_client, tools=[],
        )

        with pytest.raises(httpx.HTTPStatusError):
            # Consume all events to trigger the error
            list(session.send("hi"))
