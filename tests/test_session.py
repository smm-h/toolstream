import pytest

from openstream import AsyncSession, Result, SessionConfig, Text, ToolUse

CONFIG = SessionConfig(
    model="azure-cognitive-services/gpt-5.4-mini",
    env={
        "AZURE_API_KEY": "REDACTED_AZURE_KEY",
        "AZURE_COGNITIVE_SERVICES_RESOURCE_NAME": "normalization-playground",
    },
)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_simple_response():
    async with AsyncSession(CONFIG) as session:
        texts = []
        async for event in session.send("say hello in one word"):
            if isinstance(event, Text):
                texts.append(event.text)
        combined = "".join(texts).lower()
        assert "hello" in combined or "hi" in combined or "hey" in combined


@pytest.mark.slow
@pytest.mark.asyncio
async def test_tool_use():
    async with AsyncSession(CONFIG) as session:
        tool_events = []
        async for event in session.send("read the file /etc/hostname"):
            if isinstance(event, ToolUse):
                tool_events.append(event)
        assert len(tool_events) > 0
        assert any(e.tool == "read" for e in tool_events)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_multi_turn():
    async with AsyncSession(CONFIG) as session:
        # First turn: give it a number to remember
        async for event in session.send("remember the number 42"):
            pass

        # Second turn: ask for it back
        texts = []
        async for event in session.send("what number did I say?"):
            if isinstance(event, Text):
                texts.append(event.text)
        combined = "".join(texts)
        assert "42" in combined


@pytest.mark.slow
@pytest.mark.asyncio
async def test_cost_tracking():
    async with AsyncSession(CONFIG) as session:
        async for event in session.send("say hello"):
            pass
        assert session.total_cost > 0


@pytest.mark.slow
@pytest.mark.asyncio
async def test_result_event():
    async with AsyncSession(CONFIG) as session:
        events = []
        async for event in session.send("say hello"):
            events.append(event)
        # Result should be the last event
        assert len(events) > 0
        last = events[-1]
        assert isinstance(last, Result)
        assert last.total_input_tokens > 0
        assert last.total_output_tokens > 0
        assert last.steps >= 1
