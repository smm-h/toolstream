"""Tests for toolstream._tools -- tool registration and discovery."""

from __future__ import annotations

import types

import pytest

from toolstream._tools import Tool, collect_tools, tool


# ============================================================
# @tool decorator -- metadata storage
# ============================================================


class TestToolMetadata:
    def test_stores_tool_on_function(self):
        @tool()
        def greet(name: str) -> str:
            """Say hello."""

        assert hasattr(greet, "_tool")
        assert isinstance(greet._tool, Tool)

    def test_correct_name(self):
        @tool()
        def my_func(x: int):
            """Do something."""

        assert my_func._tool.name == "my_func"

    def test_correct_description_from_docstring(self):
        @tool()
        def my_func(x: int):
            """First line of doc.

            More details here.
            """

        assert my_func._tool.description == "First line of doc."

    def test_no_server_field(self):
        @tool()
        def fn(x: int):
            """Doc."""

        assert not hasattr(fn._tool, "server")

    def test_correct_input_schema(self):
        @tool()
        def fn(name: str, count: int = 1):
            """Doc."""

        schema = fn._tool.input_schema
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]
        assert schema["required"] == ["name"]

    def test_handler_is_original_function(self):
        @tool()
        def fn(x: int):
            """Doc."""

        assert fn._tool.handler is fn

    def test_inject_stored(self):
        @tool(inject=["ctx", "db"])
        def fn(ctx, db, name: str):
            """Doc."""

        assert fn._tool.inject == ["ctx", "db"]

    def test_no_inject_stored_as_empty_list(self):
        @tool()
        def fn(x: int):
            """Doc."""

        assert fn._tool.inject == []


# ============================================================
# @tool decorator -- identity preservation
# ============================================================


class TestToolIdentity:
    def test_returns_original_function(self):
        def fn(x: int):
            """Doc."""

        fn_before = fn
        fn_after = tool()(fn)
        assert fn_before is fn_after

    def test_function_still_callable(self):
        @tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add(2, 3) == 5


# ============================================================
# @tool decorator -- inject validation
# ============================================================


class TestToolInjectValidation:
    def test_rejects_nonexistent_inject_param(self):
        with pytest.raises(ValueError, match="inject parameter 'missing'"):

            @tool(inject=["missing"])
            def fn(x: int):
                """Doc."""

    def test_rejects_one_bad_param_among_valid(self):
        with pytest.raises(ValueError, match="inject parameter 'bad'"):

            @tool(inject=["ctx", "bad"])
            def fn(ctx, x: int):
                """Doc."""

    def test_accepts_valid_inject_param(self):
        # Should not raise
        @tool(inject=["ctx"])
        def fn(ctx, x: int):
            """Doc."""

        assert fn._tool.inject == ["ctx"]


# ============================================================
# @tool decorator -- inject excludes params from schema
# ============================================================


class TestToolInjectSchema:
    def test_injected_params_excluded_from_schema(self):
        @tool(inject=["ctx"])
        def fn(ctx, name: str, age: int):
            """Doc."""

        schema = fn._tool.input_schema
        assert "ctx" not in schema["properties"]
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_multiple_injected_params_excluded(self):
        @tool(inject=["ctx", "db"])
        def fn(ctx, db, value: str):
            """Doc."""

        schema = fn._tool.input_schema
        assert set(schema["properties"].keys()) == {"value"}


# ============================================================
# @tool decorator -- name and description defaults/overrides
# ============================================================


class TestToolNameDescription:
    def test_name_defaults_to_function_name(self):
        @tool()
        def my_cool_func(x: int):
            """Doc."""

        assert my_cool_func._tool.name == "my_cool_func"

    def test_explicit_name_overrides(self):
        @tool(name="custom_name")
        def fn(x: int):
            """Doc."""

        assert fn._tool.name == "custom_name"

    def test_description_defaults_to_first_docstring_line(self):
        @tool()
        def fn(x: int):
            """This is the summary.

            Extended description here.
            """

        assert fn._tool.description == "This is the summary."

    def test_explicit_description_overrides(self):
        @tool(description="My custom description")
        def fn(x: int):
            """This docstring is ignored."""

        assert fn._tool.description == "My custom description"

    def test_no_docstring_gives_empty_description(self):
        @tool()
        def fn(x: int):
            pass

        assert fn._tool.description == ""

    def test_empty_docstring_gives_empty_description(self):
        @tool()
        def fn(x: int):
            """"""

        assert fn._tool.description == ""


# ============================================================
# @tool decorator -- no inject
# ============================================================


class TestToolNoInject:
    def test_no_inject_all_params_in_schema(self):
        @tool()
        def fn(a: str, b: int, c: float = 0.0):
            """Doc."""

        schema = fn._tool.input_schema
        assert set(schema["properties"].keys()) == {"a", "b", "c"}
        assert set(schema["required"]) == {"a", "b"}


# ============================================================
# collect_tools -- discovery
# ============================================================


def _make_module(name: str, **attrs) -> types.ModuleType:
    """Helper to create a fake module with given attributes."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


class TestCollectTools:
    def test_finds_decorated_functions(self):
        @tool()
        def func_a(x: int):
            """A."""

        @tool()
        def func_b(y: str):
            """B."""

        mod = _make_module("mod1", func_a=func_a, func_b=func_b)
        tools = collect_tools(mod)
        names = {t.name for t in tools}
        assert names == {"func_a", "func_b"}

    def test_ignores_non_decorated(self):
        @tool()
        def decorated(x: int):
            """Doc."""

        def plain(x: int):
            """Not a tool."""

        mod = _make_module("mod1", decorated=decorated, plain=plain)
        tools = collect_tools(mod)
        assert len(tools) == 1
        assert tools[0].name == "decorated"

    def test_across_multiple_modules(self):
        @tool()
        def func_a(x: int):
            """A."""

        @tool()
        def func_b(y: str):
            """B."""

        mod1 = _make_module("mod1", func_a=func_a)
        mod2 = _make_module("mod2", func_b=func_b)
        tools = collect_tools(mod1, mod2)
        names = {t.name for t in tools}
        assert names == {"func_a", "func_b"}

    def test_raises_on_name_collision(self):
        @tool(name="shared_name")
        def func_a(x: int):
            """A."""

        @tool(name="shared_name")
        def func_b(y: str):
            """B."""

        mod1 = _make_module("mod1", func_a=func_a)
        mod2 = _make_module("mod2", func_b=func_b)

        with pytest.raises(ValueError, match="Tool name collision.*shared_name"):
            collect_tools(mod1, mod2)

    def test_name_collision_within_single_module(self):
        @tool(name="dup")
        def func_a(x: int):
            """A."""

        @tool(name="dup")
        def func_b(y: str):
            """B."""

        mod = _make_module("mod1", func_a=func_a, func_b=func_b)
        with pytest.raises(ValueError, match="Tool name collision.*dup"):
            collect_tools(mod)

    def test_empty_modules(self):
        mod = _make_module("empty")
        tools = collect_tools(mod)
        assert tools == []

    def test_no_modules(self):
        tools = collect_tools()
        assert tools == []

    def test_ignores_non_callable_with_tool_attr(self):
        # Edge case: an object that has _tool but is not callable
        class FakeObj:
            _tool = "not a real tool"

        mod = _make_module("mod1", fake=FakeObj())
        tools = collect_tools(mod)
        assert tools == []


# ============================================================
# Tool dataclass -- frozen
# ============================================================


class TestToolFrozen:
    def test_tool_is_frozen(self):
        @tool()
        def fn(x: int):
            """Doc."""

        t = fn._tool
        with pytest.raises(AttributeError):
            t.name = "new_name"
