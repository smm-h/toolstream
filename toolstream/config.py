from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._agent import AgentSandbox
    from ._context import ToolContext
    from ._tools import Tool


@dataclass
class SessionConfig:
    model: str
    api_key: str
    base_url: str
    system_prompt: str
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    agent: str | None = None
    tools: list[Tool] | None = None
    tool_context: ToolContext | None = None
    tool_env: dict[str, str] = field(default_factory=dict)
    max_completion_tokens: int = 16384
    sandbox: AgentSandbox | None = None
    metadata: dict[str, str] | None = None
    auth_style: str | None = None
    history_strategy: object | None = None
