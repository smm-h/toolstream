"""Tests for the direct Azure OpenAI backend."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from openstream._direct import DirectClient, _strip_provider

from openstream import AsyncSession, Result, SessionConfig, StepFinish, Text, ToolUse

# --- Unit tests for _strip_provider ---


def test_strip_provider_with_prefix():
    assert _strip_provider("azure-cognitive-services/gpt-5.4") == "gpt-5.4"


def test_strip_provider_no_prefix():
    assert _strip_provider("gpt-5.4") == "gpt-5.4"


def test_strip_provider_multiple_slashes():
    assert _strip_provider("a/b/gpt-5.4") == "gpt-5.4"


# --- Unit tests for tool implementations ---


@pytest.fixture
def client(tmp_path: Path) -> DirectClient:
    """Create a DirectClient pointed at a temp directory (no real API calls)."""
    config = SessionConfig(
        model="gpt-5.4",
        backend="direct",
        azure_api_key="test-key",
        azure_resource="test-resource",
        cwd=str(tmp_path),
    )
    return DirectClient(config)


@pytest.mark.asyncio
async def test_tool_read(client: DirectClient, tmp_path: Path):
    test_file = tmp_path / "hello.txt"
    test_file.write_text("line one\nline two\nline three\n")
    result = await client._tool_read(str(test_file))
    assert "1: line one" in result
    assert "2: line two" in result
    assert "3: line three" in result


@pytest.mark.asyncio
async def test_tool_read_offset_limit(client: DirectClient, tmp_path: Path):
    test_file = tmp_path / "numbers.txt"
    test_file.write_text("\n".join(f"line {i}" for i in range(10)))
    result = await client._tool_read(str(test_file), offset=2, limit=3)
    assert "3: line 2" in result
    assert "4: line 3" in result
    assert "5: line 4" in result
    assert "1: line 0" not in result
    assert "6: line 5" not in result


@pytest.mark.asyncio
async def test_tool_read_relative_path(client: DirectClient, tmp_path: Path):
    test_file = tmp_path / "relative.txt"
    test_file.write_text("content here\n")
    result = await client._tool_read("relative.txt")
    assert "1: content here" in result


@pytest.mark.asyncio
async def test_tool_write(client: DirectClient, tmp_path: Path):
    target = tmp_path / "output.txt"
    result = await client._tool_write(str(target), "hello world")
    assert "Wrote" in result
    assert target.read_text() == "hello world"


@pytest.mark.asyncio
async def test_tool_write_creates_dirs(client: DirectClient, tmp_path: Path):
    target = tmp_path / "sub" / "dir" / "file.txt"
    await client._tool_write(str(target), "nested")
    assert target.read_text() == "nested"


@pytest.mark.asyncio
async def test_tool_bash(client: DirectClient):
    result = await client._tool_bash("echo hello")
    assert "hello" in result


@pytest.mark.asyncio
async def test_tool_bash_timeout(client: DirectClient):
    result = await client._tool_bash("sleep 999", timeout=1)
    assert "timed out" in result


@pytest.mark.asyncio
async def test_tool_edit(client: DirectClient, tmp_path: Path):
    test_file = tmp_path / "edit_me.txt"
    test_file.write_text("foo bar baz")
    result = await client._tool_edit(str(test_file), "bar", "qux")
    assert "Edited" in result
    assert test_file.read_text() == "foo qux baz"


@pytest.mark.asyncio
async def test_tool_edit_not_found(client: DirectClient, tmp_path: Path):
    test_file = tmp_path / "edit_me2.txt"
    test_file.write_text("foo bar baz")
    result = await client._tool_edit(str(test_file), "nonexistent", "qux")
    assert "Error" in result


@pytest.mark.asyncio
async def test_tool_grep(client: DirectClient, tmp_path: Path):
    f1 = tmp_path / "a.py"
    f1.write_text("def hello():\n    pass\n")
    f2 = tmp_path / "b.py"
    f2.write_text("def goodbye():\n    pass\n")
    result = await client._tool_grep("hello", path=str(tmp_path))
    assert "hello" in result
    assert "goodbye" not in result


@pytest.mark.asyncio
async def test_tool_glob(client: DirectClient, tmp_path: Path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    result = await client._tool_glob(str(tmp_path / "*.py"))
    assert "a.py" in result
    assert "b.py" in result
    assert "c.txt" not in result


@pytest.mark.asyncio
async def test_truncate_output(client: DirectClient):
    # Generate output larger than 50K chars
    result = await client._tool_bash("python3 -c \"print('x' * 60000)\"")
    assert "truncated" in result
    assert len(result) < 60000


@pytest.mark.asyncio
async def test_tool_read_error(client: DirectClient):
    result = await client._execute_tool("read", {"file_path": "/nonexistent/path"})
    assert "Error" in result


@pytest.mark.asyncio
async def test_unknown_tool(client: DirectClient):
    result = await client._execute_tool("nonexistent_tool", {})
    assert "unknown tool" in result


# --- Config validation tests ---


def test_direct_backend_requires_api_key():
    with pytest.raises(ValueError, match="azure_api_key"):
        DirectClient(SessionConfig(
            model="gpt-5.4",
            backend="direct",
            azure_resource="test",
        ))


def test_direct_backend_requires_resource():
    with pytest.raises(ValueError, match="azure_resource"):
        DirectClient(SessionConfig(
            model="gpt-5.4",
            backend="direct",
            azure_api_key="test",
        ))


# --- Integration tests (require real Azure API) ---


def _load_env() -> dict[str, str]:
    """Load .env from the shopkeep root."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    env: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _make_direct_config() -> SessionConfig:
    """Create a direct-backend SessionConfig from .env."""
    env = _load_env()
    api_key = env.get("SHOPKEEP_AZURE_API_KEY") or os.environ.get("SHOPKEEP_AZURE_API_KEY")
    resource = env.get("SHOPKEEP_AZURE_RESOURCE") or os.environ.get("SHOPKEEP_AZURE_RESOURCE")
    if not api_key or not resource:
        pytest.skip("SHOPKEEP_AZURE_API_KEY and SHOPKEEP_AZURE_RESOURCE required")
    return SessionConfig(
        model="azure-cognitive-services/gpt-5.4",
        backend="direct",
        azure_api_key=api_key,
        azure_resource=resource,
    )


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
