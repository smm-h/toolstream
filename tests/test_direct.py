"""Tests for the direct LLM API backend (via AI Gateway)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from llmloop._direct import DirectClient, _strip_provider
from llmloop._tools import Tool, tool

from llmloop import AsyncSession, Result, SessionConfig, StepFinish, Text, ToolUse

# --- Unit tests for _strip_provider ---


def test_strip_provider_with_prefix():
    assert _strip_provider("azure-cognitive-services/gpt-5.4") == "gpt-5.4"


def test_strip_provider_no_prefix():
    assert _strip_provider("gpt-5.4") == "gpt-5.4"


def test_strip_provider_multiple_slashes():
    assert _strip_provider("a/b/gpt-5.4") == "gpt-5.4"


# --- Fixtures ---


def _base_config(tmp_path: Path) -> SessionConfig:
    return SessionConfig(
        model="gpt-5.4",
        api_key="test-key",
        base_url="https://test-gateway.example.com",
        system_prompt="You are a test assistant.",
        cwd=str(tmp_path),
    )


@pytest.fixture
def client(tmp_path: Path) -> DirectClient:
    """Create a DirectClient pointed at a temp directory (no real API calls)."""
    config = _base_config(tmp_path)
    return DirectClient(config)


# --- Tool dispatch tests (via _dispatch_tool) ---


@pytest.mark.asyncio
async def test_dispatch_read(client: DirectClient, tmp_path: Path):
    test_file = tmp_path / "hello.txt"
    test_file.write_text("line one\nline two\nline three\n")
    result = await client._dispatch_tool("read", {"file_path": str(test_file)})
    assert "1: line one" in result
    assert "2: line two" in result
    assert "3: line three" in result


@pytest.mark.asyncio
async def test_dispatch_read_offset_limit(client: DirectClient, tmp_path: Path):
    test_file = tmp_path / "numbers.txt"
    test_file.write_text("\n".join(f"line {i}" for i in range(10)))
    result = await client._dispatch_tool(
        "read", {"file_path": str(test_file), "offset": 2, "limit": 3},
    )
    assert "3: line 2" in result
    assert "4: line 3" in result
    assert "5: line 4" in result
    assert "1: line 0" not in result
    assert "6: line 5" not in result


@pytest.mark.asyncio
async def test_dispatch_read_relative_path(client: DirectClient, tmp_path: Path):
    test_file = tmp_path / "relative.txt"
    test_file.write_text("content here\n")
    result = await client._dispatch_tool("read", {"file_path": "relative.txt"})
    assert "1: content here" in result


@pytest.mark.asyncio
async def test_dispatch_write(client: DirectClient, tmp_path: Path):
    target = tmp_path / "output.txt"
    result = await client._dispatch_tool(
        "write", {"file_path": str(target), "content": "hello world"},
    )
    assert "Wrote" in result
    assert target.read_text() == "hello world"


@pytest.mark.asyncio
async def test_dispatch_write_creates_dirs(client: DirectClient, tmp_path: Path):
    target = tmp_path / "sub" / "dir" / "file.txt"
    await client._dispatch_tool(
        "write", {"file_path": str(target), "content": "nested"},
    )
    assert target.read_text() == "nested"


@pytest.mark.asyncio
async def test_dispatch_bash(client: DirectClient):
    result = await client._dispatch_tool("bash", {"command": "echo hello"})
    assert "hello" in result


@pytest.mark.asyncio
async def test_dispatch_bash_timeout(client: DirectClient):
    result = await client._dispatch_tool("bash", {"command": "sleep 999", "timeout": 1})
    assert "timed out" in result


@pytest.mark.asyncio
async def test_dispatch_edit(client: DirectClient, tmp_path: Path):
    test_file = tmp_path / "edit_me.txt"
    test_file.write_text("foo bar baz")
    result = await client._dispatch_tool(
        "edit",
        {"file_path": str(test_file), "old_string": "bar", "new_string": "qux"},
    )
    assert "Edited" in result
    assert test_file.read_text() == "foo qux baz"


@pytest.mark.asyncio
async def test_dispatch_edit_not_found(client: DirectClient, tmp_path: Path):
    test_file = tmp_path / "edit_me2.txt"
    test_file.write_text("foo bar baz")
    result = await client._dispatch_tool(
        "edit",
        {"file_path": str(test_file), "old_string": "nonexistent", "new_string": "qux"},
    )
    assert "Error" in result


@pytest.mark.asyncio
async def test_dispatch_grep(client: DirectClient, tmp_path: Path):
    f1 = tmp_path / "a.py"
    f1.write_text("def hello():\n    pass\n")
    f2 = tmp_path / "b.py"
    f2.write_text("def goodbye():\n    pass\n")
    result = await client._dispatch_tool(
        "grep", {"pattern": "hello", "path": str(tmp_path)},
    )
    assert "hello" in result
    assert "goodbye" not in result


@pytest.mark.asyncio
async def test_dispatch_glob(client: DirectClient, tmp_path: Path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    result = await client._dispatch_tool("glob", {"pattern": str(tmp_path / "*.py")})
    assert "a.py" in result
    assert "b.py" in result
    assert "c.txt" not in result


@pytest.mark.asyncio
async def test_dispatch_truncate_output(client: DirectClient):
    # Generate output larger than 50K chars
    result = await client._dispatch_tool(
        "bash", {"command": "python3 -c \"print('x' * 60000)\""},
    )
    assert "truncated" in result
    assert len(result) < 60000


@pytest.mark.asyncio
async def test_dispatch_read_error(client: DirectClient):
    result = await client._dispatch_tool("read", {"file_path": "/nonexistent/path"})
    assert "Error" in result


@pytest.mark.asyncio
async def test_dispatch_unknown_tool(client: DirectClient):
    result = await client._dispatch_tool("nonexistent_tool", {})
    assert "unknown tool" in result


# --- Tool exception handling ---


@pytest.mark.asyncio
async def test_dispatch_exception_becomes_error_string(tmp_path: Path):
    """When a tool handler raises, _dispatch_tool returns an error string."""

    async def exploding_handler(**kwargs: object) -> str:
        raise RuntimeError("kaboom")

    boom_tool = Tool(
        name="boom",
        description="A tool that always fails",
        input_schema={"type": "object", "properties": {}},
        handler=exploding_handler,
        inject=[],
    )
    config = _base_config(tmp_path)
    client = DirectClient(config, tools=[boom_tool])
    result = await client._dispatch_tool("boom", {})
    assert "Error: kaboom" in result


# --- Custom tool registration ---


@pytest.mark.asyncio
async def test_custom_tool_registration(tmp_path: Path):
    """User-provided tools are available for dispatch."""

    async def my_tool(x: int) -> str:
        return f"got {x}"

    custom = Tool(
        name="my_custom",
        description="custom tool",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        handler=my_tool,
        inject=[],
    )
    config = _base_config(tmp_path)
    client = DirectClient(config, tools=[custom])
    assert "my_custom" in client._tools
    result = await client._dispatch_tool("my_custom", {"x": 42})
    assert result == "got 42"


# --- User tool overrides built-in ---


@pytest.mark.asyncio
async def test_user_tool_overrides_builtin(tmp_path: Path):
    """A user tool with the same name as a built-in replaces the built-in."""

    async def custom_read(**kwargs: object) -> str:
        return "custom read result"

    override = Tool(
        name="read",
        description="overridden read",
        input_schema={"type": "object", "properties": {}},
        handler=custom_read,
        inject=[],
    )
    config = _base_config(tmp_path)
    client = DirectClient(config, tools=[override])
    # The tool should be the custom one, not the built-in
    assert client._tools["read"].description == "overridden read"
    result = await client._dispatch_tool("read", {})
    assert result == "custom read result"


# --- Tool definitions generated from Tool objects ---


def test_tool_definitions_generated(tmp_path: Path):
    """self._tool_definitions should contain entries for all 6 built-in tools."""
    config = _base_config(tmp_path)
    client = DirectClient(config)
    names = {d["function"]["name"] for d in client._tool_definitions}
    assert names == {"read", "write", "bash", "edit", "grep", "glob"}
    # Each entry should have the correct structure
    for d in client._tool_definitions:
        assert d["type"] == "function"
        assert "name" in d["function"]
        assert "description" in d["function"]
        assert "parameters" in d["function"]


def test_tool_definitions_include_custom(tmp_path: Path):
    """Custom tools appear in the generated tool definitions."""
    custom = Tool(
        name="my_custom",
        description="custom tool",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        handler=lambda: None,
        inject=[],
    )
    config = _base_config(tmp_path)
    client = DirectClient(config, tools=[custom])
    names = {d["function"]["name"] for d in client._tool_definitions}
    assert "my_custom" in names
    # Built-ins still present
    assert "read" in names


# --- HTTP client reuse ---


def test_http_client_reused(tmp_path: Path):
    """When http_client is provided, DirectClient uses it."""
    mock_client = httpx.AsyncClient()
    config = _base_config(tmp_path)
    client = DirectClient(config, http_client=mock_client)
    assert client._client is mock_client
    assert not client._owns_client


def test_http_client_created_when_none(tmp_path: Path):
    """When http_client is None, DirectClient creates its own."""
    config = _base_config(tmp_path)
    client = DirectClient(config)
    assert client._client is not None
    assert client._owns_client


# --- close() and async context manager ---


@pytest.mark.asyncio
async def test_close_closes_owned_client(tmp_path: Path):
    """close() calls aclose() on the owned HTTP client."""
    config = _base_config(tmp_path)
    client = DirectClient(config)
    assert not client._client.is_closed
    await client.close()
    assert client._client.is_closed


@pytest.mark.asyncio
async def test_close_does_not_close_injected_client(tmp_path: Path):
    """close() does NOT close an injected HTTP client."""
    injected = httpx.AsyncClient()
    config = _base_config(tmp_path)
    client = DirectClient(config, http_client=injected)
    await client.close()
    assert not injected.is_closed
    # Clean up the injected client ourselves
    await injected.aclose()


@pytest.mark.asyncio
async def test_async_context_manager(tmp_path: Path):
    """DirectClient can be used as an async context manager."""
    config = _base_config(tmp_path)
    async with DirectClient(config) as client:
        assert not client._client.is_closed
        assert client.session_id  # sanity check
    assert client._client.is_closed


# --- max_completion_tokens ---


def test_max_completion_tokens_default(tmp_path: Path):
    config = _base_config(tmp_path)
    client = DirectClient(config)
    assert client._max_completion_tokens == 16384


def test_max_completion_tokens_override(tmp_path: Path):
    config = _base_config(tmp_path)
    client = DirectClient(config, max_completion_tokens=4096)
    assert client._max_completion_tokens == 4096


# --- Config validation tests ---


def test_direct_backend_requires_api_key():
    with pytest.raises(ValueError, match="api_key"):
        DirectClient(SessionConfig(
            model="gpt-5.4",
            api_key="",
            base_url="https://test-gateway.example.com",
            system_prompt="test",
        ))


def test_direct_backend_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        DirectClient(SessionConfig(
            model="gpt-5.4",
            api_key="test",
            base_url="",
            system_prompt="test",
        ))


# --- Integration tests (require real Azure API) ---


_TEST_API_KEY = os.environ.get("LLMLOOP_TEST_API_KEY", "")
_TEST_BASE_URL = os.environ.get("LLMLOOP_TEST_BASE_URL", "")
_HAS_CREDENTIALS = bool(_TEST_API_KEY and _TEST_BASE_URL)


def _make_direct_config() -> SessionConfig:
    """Create a direct-backend SessionConfig from environment variables."""
    return SessionConfig(
        model="gpt-5.4",
        api_key=_TEST_API_KEY,
        base_url=_TEST_BASE_URL,
        system_prompt="You are a helpful coding assistant.",
    )


@pytest.mark.skipif(not _HAS_CREDENTIALS, reason="LLMLOOP_TEST_API_KEY and LLMLOOP_TEST_BASE_URL required")
@pytest.mark.slow
@pytest.mark.asyncio
async def test_direct_simple():
    config = _make_direct_config()
    async with AsyncSession(config) as session:
        texts: list[str] = []
        async for event in session.send("say hello in one word"):
            if isinstance(event, Text):
                texts.append(event.text)
        combined = "".join(texts).lower()
        assert "hello" in combined or "hi" in combined or "hey" in combined


@pytest.mark.skipif(not _HAS_CREDENTIALS, reason="LLMLOOP_TEST_API_KEY and LLMLOOP_TEST_BASE_URL required")
@pytest.mark.slow
@pytest.mark.asyncio
async def test_direct_tool_use():
    config = _make_direct_config()
    async with AsyncSession(config) as session:
        tool_events: list[ToolUse] = []
        async for event in session.send("read the file /etc/hostname"):
            if isinstance(event, ToolUse):
                tool_events.append(event)
        assert len(tool_events) > 0
        assert any(e.tool == "read" for e in tool_events)


@pytest.mark.skipif(not _HAS_CREDENTIALS, reason="LLMLOOP_TEST_API_KEY and LLMLOOP_TEST_BASE_URL required")
@pytest.mark.slow
@pytest.mark.asyncio
async def test_direct_multi_turn():
    config = _make_direct_config()
    async with AsyncSession(config) as session:
        # First turn
        async for event in session.send("remember the number 42"):
            pass

        # Second turn
        texts: list[str] = []
        async for event in session.send("what number did I say?"):
            if isinstance(event, Text):
                texts.append(event.text)
        combined = "".join(texts)
        assert "42" in combined


@pytest.mark.skipif(not _HAS_CREDENTIALS, reason="LLMLOOP_TEST_API_KEY and LLMLOOP_TEST_BASE_URL required")
@pytest.mark.slow
@pytest.mark.asyncio
async def test_direct_result_event():
    config = _make_direct_config()
    async with AsyncSession(config) as session:
        events = []
        async for event in session.send("say hello"):
            events.append(event)
        assert len(events) > 0
        last = events[-1]
        assert isinstance(last, Result)
        assert last.total_input_tokens > 0
        assert last.total_output_tokens > 0
        assert last.steps >= 1


@pytest.mark.skipif(not _HAS_CREDENTIALS, reason="LLMLOOP_TEST_API_KEY and LLMLOOP_TEST_BASE_URL required")
@pytest.mark.slow
@pytest.mark.asyncio
async def test_direct_step_finish_tokens():
    config = _make_direct_config()
    async with AsyncSession(config) as session:
        step_finishes: list[StepFinish] = []
        async for event in session.send("say hello"):
            if isinstance(event, StepFinish):
                step_finishes.append(event)
        assert len(step_finishes) > 0
        # The final StepFinish should have token counts
        final = step_finishes[-1]
        assert final.input_tokens > 0
        assert final.output_tokens > 0
        assert final.reason == "stop"
