"""Tests for llmloop._invoke -- agent invocation helpers."""

from __future__ import annotations

import pytest

from llmloop._agent import AgentDefinition, ToolRef
from llmloop._invoke import (
    _build_invocation_config,
    _filter_tools,
    invoke_agent,
    invoke_agent_sync,
)
from llmloop._session import AsyncSession, SyncSession
from llmloop._tools import Tool, tool
from llmloop.config import SessionConfig


# ============================================================
# Helpers
# ============================================================


def _make_definition(
    *,
    name="test-agent",
    prompt_template="You are a test agent.",
    version="1.0",
    tools=None,
    model=None,
):
    return AgentDefinition(
        name=name,
        prompt_template=prompt_template,
        version=version,
        tools=tools,
        model=model,
    )


def _make_config(**overrides):
    defaults = dict(model="base-model", backend="opencode", cwd="/tmp")
    defaults.update(overrides)
    return SessionConfig(**defaults)


# ============================================================
# TestFilterTools -- _filter_tools()
# ============================================================


class TestFilterTools:
    def test_none_definition_tools_returns_none(self):
        """definition.tools=None -> returns None regardless of available_tools."""

        def bash_handler(command: str) -> str:
            """Run a command."""

        definition = _make_definition(tools=None)
        result = _filter_tools(definition, {"bash": bash_handler})
        assert result is None

    def test_none_available_tools_returns_none(self):
        """available_tools=None -> returns None regardless of definition.tools."""
        definition = _make_definition(tools=[ToolRef("bash")])
        result = _filter_tools(definition, None)
        assert result is None

    def test_both_none_returns_none(self):
        """Both definition.tools and available_tools are None -> returns None."""
        definition = _make_definition(tools=None)
        result = _filter_tools(definition, None)
        assert result is None

    def test_filters_to_declared_tools_only(self):
        """Only tools declared in the definition are included, extras are ignored."""

        def bash_handler(command: str) -> str:
            """Run a command."""

        def fetch_handler(url: str) -> str:
            """Fetch a URL."""

        def navigate_handler(url: str) -> str:
            """Navigate to a URL."""

        definition = _make_definition(tools=[ToolRef("bash"), ToolRef("fetch")])
        available = {
            "bash": bash_handler,
            "fetch": fetch_handler,
            "navigate": navigate_handler,
        }
        result = _filter_tools(definition, available)

        assert result is not None
        assert len(result) == 2
        names = {t.name for t in result}
        assert names == {"bash", "fetch"}

    def test_missing_handler_raises_value_error(self):
        """A declared tool with no matching handler raises ValueError."""

        def bash_handler(command: str) -> str:
            """Run a command."""

        definition = _make_definition(tools=[ToolRef("nonexistent")])
        with pytest.raises(ValueError, match="nonexistent"):
            _filter_tools(definition, {"bash": bash_handler})

    def test_multiple_missing_handlers_listed(self):
        """All missing handler names appear in the error message."""
        definition = _make_definition(tools=[ToolRef("foo"), ToolRef("bar")])
        with pytest.raises(ValueError, match="foo") as exc_info:
            _filter_tools(definition, {})
        assert "bar" in str(exc_info.value)

    def test_tool_decorated_handler_uses_precomputed(self):
        """A @tool-decorated handler reuses its precomputed _tool object."""

        @tool("my-server")
        def decorated_fn(x: int) -> str:
            """A decorated tool."""

        definition = _make_definition(tools=[ToolRef("decorated_fn")])
        result = _filter_tools(definition, {"decorated_fn": decorated_fn})

        assert result is not None
        assert len(result) == 1
        assert result[0] is decorated_fn._tool

    def test_plain_handler_gets_auto_schema(self):
        """A plain function gets a Tool with server='dynamic', inject=[], and correct schema."""

        def my_plain(name: str, count: int = 1) -> str:
            """Do something."""

        definition = _make_definition(tools=[ToolRef("my_plain")])
        result = _filter_tools(definition, {"my_plain": my_plain})

        assert result is not None
        assert len(result) == 1
        t = result[0]
        assert t.name == "my_plain"
        assert t.server == "dynamic"
        assert t.inject == []
        assert t.input_schema["type"] == "object"
        assert "name" in t.input_schema["properties"]
        assert "count" in t.input_schema["properties"]

    def test_plain_handler_description_from_docstring(self):
        """A plain function's description comes from its docstring's first line."""

        def documented(x: int) -> str:
            """First line of docs.

            More details here.
            """

        definition = _make_definition(tools=[ToolRef("documented")])
        result = _filter_tools(definition, {"documented": documented})

        assert result is not None
        assert result[0].description == "First line of docs."

    def test_plain_handler_no_docstring_empty_description(self):
        """A plain function with no docstring gets an empty description."""

        def undocumented(x: int) -> str:
            pass

        definition = _make_definition(tools=[ToolRef("undocumented")])
        result = _filter_tools(definition, {"undocumented": undocumented})

        assert result is not None
        assert result[0].description == ""

    def test_mixed_decorated_and_plain(self):
        """One @tool-decorated and one plain handler both resolve correctly."""

        @tool("srv")
        def decorated(x: int) -> str:
            """Decorated tool."""

        def plain(y: str) -> str:
            """Plain handler."""

        definition = _make_definition(
            tools=[ToolRef("decorated"), ToolRef("plain")]
        )
        result = _filter_tools(
            definition, {"decorated": decorated, "plain": plain}
        )

        assert result is not None
        assert len(result) == 2

        by_name = {t.name: t for t in result}
        assert by_name["decorated"] is decorated._tool
        assert by_name["plain"].server == "dynamic"
        assert by_name["plain"].description == "Plain handler."


# ============================================================
# TestBuildInvocationConfig -- _build_invocation_config()
# ============================================================


class TestBuildInvocationConfig:
    def test_prompt_resolution(self):
        """Prompt template variables are substituted in the result."""
        definition = _make_definition(prompt_template="Hello {name}")
        config = _make_config()
        result = _build_invocation_config(
            definition, config, variables={"name": "world"}
        )
        assert result.system_prompt == "Hello world"

    def test_prompt_unresolved_variable_raises(self):
        """Missing template variable raises ValueError from resolve_prompt."""
        definition = _make_definition(prompt_template="{missing}")
        config = _make_config()
        with pytest.raises(ValueError, match="missing"):
            _build_invocation_config(definition, config, variables={})

    def test_model_from_definition_wins(self):
        """When definition sets a model, it overrides the config model."""
        definition = _make_definition(
            prompt_template="prompt", model="def-model"
        )
        config = _make_config(model="cfg-model")
        result = _build_invocation_config(definition, config)
        assert result.model == "def-model"

    def test_model_falls_back_to_config(self):
        """When definition.model is None, config.model is used."""
        definition = _make_definition(prompt_template="prompt", model=None)
        config = _make_config(model="cfg-model")
        result = _build_invocation_config(definition, config)
        assert result.model == "cfg-model"

    def test_config_fields_carried_through(self):
        """backend, cwd, api_key, base_url, tool_context, max_completion_tokens are preserved."""
        ctx = object()
        config = _make_config(
            backend="direct",
            cwd="/some/path",
            api_key="sk-test",
            base_url="https://api.example.com",
            tool_context=ctx,
            max_completion_tokens=8192,
        )
        definition = _make_definition(prompt_template="prompt")
        result = _build_invocation_config(definition, config)

        assert result.backend == "direct"
        assert result.cwd == "/some/path"
        assert result.api_key == "sk-test"
        assert result.base_url == "https://api.example.com"
        assert result.tool_context is ctx
        assert result.max_completion_tokens == 8192

    def test_tools_none_when_no_declaration(self):
        """When definition.tools is None, result.tools is None."""
        definition = _make_definition(prompt_template="prompt", tools=None)
        config = _make_config()
        result = _build_invocation_config(definition, config)
        assert result.tools is None

    def test_tools_filtered_when_declared(self):
        """When definition declares tools, result.tools is filtered correctly."""

        def handler_a(x: int) -> str:
            """A."""

        def handler_b(y: str) -> str:
            """B."""

        definition = _make_definition(
            prompt_template="prompt",
            tools=[ToolRef("a")],
        )
        config = _make_config()
        result = _build_invocation_config(
            definition,
            config,
            available_tools={"a": handler_a, "b": handler_b},
        )

        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0].name == "a"


# ============================================================
# TestInvokeAgent -- invoke_agent() async context manager
# ============================================================


class TestInvokeAgent:
    @pytest.mark.asyncio
    async def test_yields_async_session(self):
        """invoke_agent yields an AsyncSession instance."""
        definition = _make_definition(
            name="test",
            prompt_template="You are {role}",
            version="1.0",
            model=None,
        )
        config = _make_config(model="test-model", backend="opencode", cwd="/tmp")

        async with invoke_agent(
            definition, config, variables={"role": "assistant"}
        ) as session:
            assert isinstance(session, AsyncSession)

    @pytest.mark.asyncio
    async def test_session_has_resolved_prompt(self):
        """The yielded session's config has the resolved system prompt."""
        definition = _make_definition(
            name="test",
            prompt_template="You are {role}",
            version="1.0",
            model=None,
        )
        config = _make_config(model="test-model", backend="opencode", cwd="/tmp")

        async with invoke_agent(
            definition, config, variables={"role": "assistant"}
        ) as session:
            assert session._config.system_prompt == "You are assistant"


# ============================================================
# TestInvokeAgentSync -- invoke_agent_sync() sync context manager
# ============================================================


class TestInvokeAgentSync:
    def test_yields_sync_session(self):
        """invoke_agent_sync yields a SyncSession instance."""
        definition = _make_definition(
            name="test",
            prompt_template="You are {role}",
            version="1.0",
            model=None,
        )
        config = _make_config(model="test-model", backend="opencode", cwd="/tmp")

        with invoke_agent_sync(
            definition, config, variables={"role": "assistant"}
        ) as session:
            assert isinstance(session, SyncSession)

    def test_sync_session_has_resolved_prompt(self):
        """The yielded sync session's config has the resolved system prompt."""
        definition = _make_definition(
            name="test",
            prompt_template="You are {role}",
            version="1.0",
            model=None,
        )
        config = _make_config(model="test-model", backend="opencode", cwd="/tmp")

        with invoke_agent_sync(
            definition, config, variables={"role": "assistant"}
        ) as session:
            assert session._config.system_prompt == "You are assistant"
