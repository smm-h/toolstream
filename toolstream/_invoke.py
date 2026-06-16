"""Agent invocation helpers.

Build a SessionConfig from an AgentDefinition and yield a ready-to-use
session (async or sync).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Callable

from ._agent import AgentDefinition, ToolRef, resolve_prompt
from ._schema import _generate_schema
from ._session import AsyncSession, SyncSession
from ._tools import Tool
from .config import SessionConfig

__all__ = ["invoke_agent", "invoke_agent_sync"]


def _filter_tools(
    definition: AgentDefinition,
    available_tools: dict[str, Callable] | None,
) -> list[Tool] | None:
    """Match ToolRefs in *definition* against *available_tools* handlers.

    Returns None when the definition declares no tools or no tools dict
    is provided.  Raises ValueError if any declared tool name is missing
    from *available_tools*.
    """
    if definition.tools is None or available_tools is None:
        return None

    missing: list[str] = []
    matched: list[Tool] = []

    for ref in definition.tools:
        handler = available_tools.get(ref.name)
        if handler is None:
            missing.append(ref.name)
            continue

        if hasattr(handler, "_tool"):
            matched.append(handler._tool)
        else:
            # Generate schema on the fly for plain callables.
            input_schema = _generate_schema(handler, inject=set())
            doc = handler.__doc__
            description = doc.strip().split("\n")[0].strip() if doc else ""
            matched.append(
                Tool(
                    name=ref.name,
                    description=description,
                    input_schema=input_schema,
                    handler=handler,
                    inject=[],
                )
            )

    if missing:
        raise ValueError(
            f"Missing tool handlers: {', '.join(missing)}"
        )

    return matched


def _build_invocation_config(
    definition: AgentDefinition,
    config: SessionConfig,
    *,
    variables: dict[str, str] | None = None,
    available_tools: dict[str, Callable] | None = None,
) -> SessionConfig:
    """Create a SessionConfig tailored to *definition*."""
    system_prompt = resolve_prompt(definition.prompt_template, variables or {})
    tools = _filter_tools(definition, available_tools)

    return SessionConfig(
        model=definition.model or config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        system_prompt=system_prompt,
        cwd=config.cwd,
        tools=tools,
        tool_context=config.tool_context,
        tool_env=config.tool_env,
        max_completion_tokens=config.max_completion_tokens,
        sandbox=config.sandbox,
        metadata=config.metadata,
    )


@contextlib.asynccontextmanager
async def invoke_agent(
    definition: AgentDefinition,
    config: SessionConfig,
    *,
    variables: dict[str, str] | None = None,
    available_tools: dict[str, Callable] | None = None,
) -> AsyncIterator[AsyncSession]:
    """Async context manager that yields an AsyncSession for *definition*."""
    invocation_config = _build_invocation_config(
        definition, config, variables=variables, available_tools=available_tools,
    )
    session = AsyncSession(invocation_config)
    async with session:
        yield session


@contextlib.contextmanager
def invoke_agent_sync(
    definition: AgentDefinition,
    config: SessionConfig,
    *,
    variables: dict[str, str] | None = None,
    available_tools: dict[str, Callable] | None = None,
) -> Iterator[SyncSession]:
    """Sync context manager that yields a SyncSession for *definition*."""
    invocation_config = _build_invocation_config(
        definition, config, variables=variables, available_tools=available_tools,
    )
    session = SyncSession(invocation_config)
    with session:
        yield session
