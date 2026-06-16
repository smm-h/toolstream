"""Tool registration system for llmloop.

Provides the @tool decorator for marking functions as LLM-callable tools,
and collect_tools() for discovering decorated functions across modules.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

from ._schema import _generate_schema


@dataclass(frozen=True)
class Tool:
    """Metadata for a registered tool function."""

    name: str
    description: str
    input_schema: dict
    handler: Callable
    server: str
    inject: list[str]


def tool(
    server: str,
    *,
    name: str | None = None,
    description: str | None = None,
    inject: list[str] | None = None,
) -> Callable:
    """Decorator factory that attaches Tool metadata to a function.

    Usage:
        @tool("my-server")
        def my_func(x: int) -> str:
            ...

        @tool("my-server", inject=["ctx"])
        def my_func(ctx, x: int) -> str:
            ...

    The decorated function is returned unchanged -- this decorator only
    attaches a _tool attribute, it does not wrap the function.
    """
    inject_list = inject or []

    def decorator(fn: Callable) -> Callable:
        # Validate that every injected param exists in the signature
        sig = inspect.signature(fn)
        for param_name in inject_list:
            if param_name not in sig.parameters:
                raise ValueError(
                    f"inject parameter {param_name!r} not found in "
                    f"signature of {fn.__name__}(). "
                    f"Available: {list(sig.parameters.keys())}"
                )

        # Resolve name
        tool_name = name if name is not None else fn.__name__

        # Resolve description from first line of docstring
        if description is not None:
            tool_description = description
        elif fn.__doc__:
            tool_description = fn.__doc__.strip().split("\n")[0].strip()
        else:
            tool_description = ""

        # Generate input schema, excluding injected params
        input_schema = _generate_schema(fn, inject=set(inject_list))

        fn._tool = Tool(
            name=tool_name,
            description=tool_description,
            input_schema=input_schema,
            handler=fn,
            server=server,
            inject=inject_list,
        )

        return fn

    return decorator


def collect_tools(*modules) -> list[Tool]:
    """Discover all @tool-decorated functions across the given modules.

    Raises ValueError if two tools share the same name.
    """
    tools: list[Tool] = []
    seen_names: dict[str, str] = {}  # name -> module name for error messages

    for module in modules:
        module_name = getattr(module, "__name__", repr(module))
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if callable(obj) and hasattr(obj, "_tool"):
                t: Tool = obj._tool
                if t.name in seen_names:
                    raise ValueError(
                        f"Tool name collision: {t.name!r} found in both "
                        f"{seen_names[t.name]} and {module_name}"
                    )
                seen_names[t.name] = module_name
                tools.append(t)

    return tools
