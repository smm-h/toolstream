"""Tests for context injection in DirectClient and PipelineContext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from openstream._context import PipelineContext
from openstream._direct import DirectClient
from openstream._tools import Tool
from openstream.config import SessionConfig


# -- helpers --

def _make_tool(
    name: str,
    handler,
    inject: list[str] | None = None,
) -> Tool:
    """Build a Tool with minimal boilerplate."""
    return Tool(
        name=name,
        description=f"test tool {name}",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
        server="test",
        inject=inject or [],
    )


def _make_config() -> SessionConfig:
    return SessionConfig(
        model="test-model",
        backend="direct",
        api_key="test-key",
        base_url="http://localhost:9999/v1/chat/completions",
    )


def _make_client(
    tools: list[Tool] | None = None,
    tool_context: object | None = None,
) -> DirectClient:
    """Build a DirectClient with a dummy HTTP client (never used in dispatch tests)."""
    return DirectClient(
        _make_config(),
        tools=tools,
        tool_context=tool_context,
        http_client=httpx.AsyncClient(),
    )


# -- injection tests --

class TestContextInjection:

    @pytest.mark.asyncio
    async def test_inject_resolves_attribute(self):
        """Tool with inject=["foo"] receives ctx.foo when tool_context has foo."""

        @dataclass
        class Ctx:
            foo: str = "hello"

        received: dict = {}

        async def handler(foo: str, x: int = 0) -> str:
            received["foo"] = foo
            received["x"] = x
            return "ok"

        tool = _make_tool("my_tool", handler, inject=["foo"])
        client = _make_client(tools=[tool], tool_context=Ctx(foo="world"))
        result = await client._dispatch_tool("my_tool", {"x": 42})

        assert result == "ok"
        assert received["foo"] == "world"
        assert received["x"] == 42

    @pytest.mark.asyncio
    async def test_inject_raises_when_tool_context_is_none(self):
        """Tool with inject=["foo"] raises RuntimeError when tool_context is None."""

        async def handler(foo: str) -> str:
            return "ok"

        tool = _make_tool("my_tool", handler, inject=["foo"])
        client = _make_client(tools=[tool], tool_context=None)

        with pytest.raises(RuntimeError, match="tool_context.*None"):
            await client._dispatch_tool("my_tool", {})

    @pytest.mark.asyncio
    async def test_inject_raises_when_attribute_missing(self):
        """Tool with inject=["foo"] raises AttributeError when tool_context lacks foo."""

        @dataclass
        class Ctx:
            bar: str = "nope"

        async def handler(foo: str) -> str:
            return "ok"

        tool = _make_tool("my_tool", handler, inject=["foo"])
        client = _make_client(tools=[tool], tool_context=Ctx())

        with pytest.raises(AttributeError, match="has no attribute 'foo'"):
            await client._dispatch_tool("my_tool", {})

    @pytest.mark.asyncio
    async def test_no_inject_works_without_context(self):
        """Tool with no inject works fine without tool_context."""

        async def handler(x: int = 0) -> str:
            return f"got {x}"

        tool = _make_tool("my_tool", handler, inject=[])
        client = _make_client(tools=[tool], tool_context=None)

        result = await client._dispatch_tool("my_tool", {"x": 7})
        assert result == "got 7"

    @pytest.mark.asyncio
    async def test_multiple_inject_params(self):
        """Multiple inject params are all resolved correctly."""

        @dataclass
        class Ctx:
            alpha: str = "a"
            beta: int = 2

        received: dict = {}

        async def handler(alpha: str, beta: int, x: str = "") -> str:
            received["alpha"] = alpha
            received["beta"] = beta
            received["x"] = x
            return "ok"

        tool = _make_tool("my_tool", handler, inject=["alpha", "beta"])
        client = _make_client(tools=[tool], tool_context=Ctx(alpha="A", beta=99))
        result = await client._dispatch_tool("my_tool", {"x": "hello"})

        assert result == "ok"
        assert received["alpha"] == "A"
        assert received["beta"] == 99
        assert received["x"] == "hello"

    @pytest.mark.asyncio
    async def test_inject_with_pipeline_context(self):
        """PipelineContext works as tool_context for injection."""

        received: dict = {}

        async def handler(browser_ctx: Any = None) -> str:
            received["browser_ctx"] = browser_ctx
            return "ok"

        fake_browser = object()
        ctx = PipelineContext(browser_ctx=fake_browser)
        tool = _make_tool("my_tool", handler, inject=["browser_ctx"])
        client = _make_client(tools=[tool], tool_context=ctx)

        result = await client._dispatch_tool("my_tool", {})
        assert result == "ok"
        assert received["browser_ctx"] is fake_browser


# -- PipelineContext tests --

class TestPipelineContext:

    def test_default_fields_are_none(self):
        ctx = PipelineContext()
        assert ctx.spawn_ctx is None
        assert ctx.browser_ctx is None

    def test_holds_sub_contexts(self):
        spawn = object()
        browser = object()
        ctx = PipelineContext(spawn_ctx=spawn, browser_ctx=browser)
        assert ctx.spawn_ctx is spawn
        assert ctx.browser_ctx is browser

    def test_getattr_returns_correct_values(self):
        """getattr on PipelineContext returns the correct sub-contexts."""
        spawn = {"model": "test"}
        browser = {"page": "mock"}
        ctx = PipelineContext(spawn_ctx=spawn, browser_ctx=browser)
        assert getattr(ctx, "spawn_ctx") is spawn
        assert getattr(ctx, "browser_ctx") is browser

    def test_getattr_missing_raises(self):
        ctx = PipelineContext()
        with pytest.raises(AttributeError):
            getattr(ctx, "nonexistent_field")
