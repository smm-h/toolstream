"""Integration tests exercising the full toolstream stack with mock LLM responder."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from dataclasses import dataclass
from unittest.mock import patch

import httpx
import pytest

from toolstream._agent import AgentDefinition, ToolRef
from toolstream._direct import DirectClient
from toolstream._invoke import invoke_agent
from toolstream._session import AsyncSession, SyncSession
from toolstream._tools import Tool, tool
from toolstream.events import Result, StepFinish, StepStart, Text, ToolUse

from .conftest import direct_config, text_response, tool_call_response


def multi_tool_call_response(
    tool_calls: list[tuple[str, dict, str]],
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict:
    """Build a canned response with multiple tool calls.

    Each item in tool_calls is (tool_name, arguments_dict, call_id).
    """
    return {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args),
                        },
                    }
                    for name, args, call_id in tool_calls
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


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

    # DirectClient emits per-step deltas in each StepFinish (not running
    # totals). The first StepFinish has 100 input / 50 output, the second
    # has 200 input / 80 output. AsyncSession._send_direct accumulates
    # these via +=, so the Result reflects the correct sum of per-step values.
    assert result.total_input_tokens == 300  # 100 + 200
    assert result.total_output_tokens == 130  # 50 + 80

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


async def test_multi_round_tool_calling_loop(mock_llm_responses):
    """Three-round tool loop: read tool, write tool, then final text."""

    async def read_handler(file_path: str) -> str:
        return "file contents"

    async def write_handler(file_path: str, content: str) -> str:
        return "written"

    read_tool = Tool(
        name="read",
        description="Read a file",
        input_schema={
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
        handler=read_handler,
        inject=[],
    )
    write_tool = Tool(
        name="write",
        description="Write a file",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
        handler=write_handler,
        inject=[],
    )

    mock_client = mock_llm_responses(
        tool_call_response(
            "read", {"file_path": "/tmp/test.txt"},
            call_id="call_read", prompt_tokens=100, completion_tokens=20,
        ),
        tool_call_response(
            "write", {"file_path": "/tmp/out.txt", "content": "hello"},
            call_id="call_write", prompt_tokens=150, completion_tokens=30,
        ),
        text_response("Done writing", prompt_tokens=200, completion_tokens=40),
    )
    config = direct_config()

    client = DirectClient(
        config, http_client=mock_client, tools=[read_tool, write_tool],
    )
    events = []
    async for event in client.send("read and write"):
        events.append(event)

    # Verify event sequence
    assert isinstance(events[0], StepStart)

    assert isinstance(events[1], ToolUse)
    assert events[1].tool == "read"
    assert events[1].call_id == "call_read"
    assert events[1].output == "file contents"
    assert events[1].input == {"file_path": "/tmp/test.txt"}

    assert isinstance(events[2], StepFinish)
    assert events[2].reason == "tool-calls"

    assert isinstance(events[3], ToolUse)
    assert events[3].tool == "write"
    assert events[3].call_id == "call_write"
    assert events[3].output == "written"
    assert events[3].input == {"file_path": "/tmp/out.txt", "content": "hello"}

    assert isinstance(events[4], StepFinish)
    assert events[4].reason == "tool-calls"

    assert isinstance(events[5], Text)
    assert events[5].text == "Done writing"

    assert isinstance(events[6], StepFinish)
    assert events[6].reason == "stop"

    # Verify exactly 7 events (no leftover mock responses consumed)
    assert len(events) == 7

    # Each StepFinish carries only its own API call's tokens (per-step deltas)
    final_step = events[6]
    assert final_step.input_tokens == 200  # just the last call
    assert final_step.output_tokens == 40  # just the last call


async def test_multi_turn_conversation(mock_llm_responses):
    """Two-turn conversation via AsyncSession: each turn gets its own Result."""
    mock_client = mock_llm_responses(
        text_response("First answer", prompt_tokens=50, completion_tokens=10),
        text_response("Second answer", prompt_tokens=80, completion_tokens=15),
    )
    config = direct_config()

    session = AsyncSession(config)
    session._direct = DirectClient(config, http_client=mock_client, tools=[])

    # Turn 1
    events_1 = []
    async for event in session.send("question 1"):
        events_1.append(event)

    assert isinstance(events_1[0], StepStart)
    assert isinstance(events_1[1], Text)
    assert events_1[1].text == "First answer"
    assert isinstance(events_1[2], StepFinish)
    assert events_1[2].reason == "stop"
    assert isinstance(events_1[3], Result)
    result_1 = events_1[3]
    assert result_1.total_input_tokens == 50
    assert result_1.total_output_tokens == 10

    # Turn 2
    events_2 = []
    async for event in session.send("question 2"):
        events_2.append(event)

    assert isinstance(events_2[0], StepStart)
    assert isinstance(events_2[1], Text)
    assert events_2[1].text == "Second answer"
    assert isinstance(events_2[2], StepFinish)
    assert events_2[2].reason == "stop"
    assert isinstance(events_2[3], Result)
    result_2 = events_2[3]

    # Turn 2 Result has its own token counts, NOT cumulative with turn 1
    assert result_2.total_input_tokens == 80
    assert result_2.total_output_tokens == 15

    await session.close()


async def test_invoke_agent_with_message(mock_llm_responses):
    """invoke_agent creates a session whose system prompt is the resolved template."""

    captured_requests: list[dict] = []

    def capturing_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        resp = text_response("Agent response")
        return httpx.Response(200, json=resp)

    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(capturing_handler),
    )

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
        # Inject mock http_client into the DirectClient
        session._direct = DirectClient(
            session._config,
            http_client=mock_client,
            tools=session._config.tools,
        )

        events = []
        async for event in session.send("hello agent"):
            events.append(event)

    # Verify the system prompt in the API call is the resolved template
    assert len(captured_requests) == 1
    messages = captured_requests[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."

    # Verify events flow correctly
    assert isinstance(events[0], StepStart)
    assert isinstance(events[1], Text)
    assert events[1].text == "Agent response"
    assert isinstance(events[2], StepFinish)
    assert events[2].reason == "stop"
    assert isinstance(events[3], Result)


async def test_multiple_tool_calls_in_single_response(mock_llm_responses):
    """LLM returns 2 tool calls in a single response; both are dispatched."""

    async def add_handler(a: int, b: int) -> str:
        return str(a + b)

    async def multiply_handler(a: int, b: int) -> str:
        return str(a * b)

    add_tool = Tool(
        name="add",
        description="Add two numbers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        handler=add_handler,
        inject=[],
    )
    multiply_tool = Tool(
        name="multiply",
        description="Multiply two numbers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        handler=multiply_handler,
        inject=[],
    )

    mock_client = mock_llm_responses(
        multi_tool_call_response([
            ("add", {"a": 3, "b": 4}, "call_add"),
            ("multiply", {"a": 3, "b": 4}, "call_mul"),
        ]),
        text_response("3+4=7 and 3*4=12"),
    )
    config = direct_config()

    client = DirectClient(
        config, http_client=mock_client, tools=[add_tool, multiply_tool],
    )
    events = []
    async for event in client.send("compute 3+4 and 3*4"):
        events.append(event)

    # First response yields 2 ToolUse events (one per tool call)
    assert isinstance(events[0], StepStart)

    assert isinstance(events[1], ToolUse)
    assert events[1].tool == "add"
    assert events[1].call_id == "call_add"
    assert events[1].output == "7"
    assert events[1].input == {"a": 3, "b": 4}

    assert isinstance(events[2], ToolUse)
    assert events[2].tool == "multiply"
    assert events[2].call_id == "call_mul"
    assert events[2].output == "12"
    assert events[2].input == {"a": 3, "b": 4}

    assert isinstance(events[3], StepFinish)
    assert events[3].reason == "tool-calls"

    # Second API call produces the final text
    assert isinstance(events[4], Text)
    assert events[4].text == "3+4=7 and 3*4=12"

    assert isinstance(events[5], StepFinish)
    assert events[5].reason == "stop"

    assert len(events) == 6

    # Verify both tool results were appended to messages by checking that
    # the final response was reached (2 mock responses consumed, no error)


def test_sync_session_keyboard_interrupt_cancels_future():
    """KeyboardInterrupt during SyncSession.send() cancels the background Future.

    Verifies that the future returned by run_coroutine_threadsafe is stored
    and cancelled when KeyboardInterrupt is raised during queue consumption.
    We mock the async session to produce events slowly, then raise
    KeyboardInterrupt via a queue wrapper that interrupts on the second get().
    """
    config = direct_config()

    with SyncSession(config) as session:
        cancelled = threading.Event()

        # Track the real future so we can verify it was stored
        real_future_ref: list = []
        original_run = asyncio.run_coroutine_threadsafe

        def tracking_run(coro, loop):
            fut = original_run(coro, loop)
            real_future_ref.append(fut)
            # Wrap the future to detect cancel() calls
            original_cancel = fut.cancel

            def tracked_cancel(*args, **kwargs):
                cancelled.set()
                return original_cancel(*args, **kwargs)

            fut.cancel = tracked_cancel
            return fut

        # Make the async session produce one event then block forever
        async def slow_send(message):
            yield "first_event"
            await asyncio.sleep(60)

        session._async_session.send = slow_send  # type: ignore[assignment]

        # Replace queue.Queue.get to raise KeyboardInterrupt on second call
        call_count = 0
        original_get = queue.Queue.get

        def interrupting_get(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise KeyboardInterrupt
            return original_get(self, *args, **kwargs)

        with patch.object(queue.Queue, "get", interrupting_get), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=tracking_run):
            with pytest.raises(KeyboardInterrupt):
                list(session.send("test"))

        assert cancelled.is_set(), "future.cancel() was not called on KeyboardInterrupt"
