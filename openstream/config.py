from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionConfig:
    model: str  # e.g. "azure-cognitive-services/gpt-5.4-mini"
    cwd: str | None = None  # working directory
    opencode_binary: str | None = None  # path to opencode binary, auto-detected if None
    env: dict[str, str] = field(default_factory=dict)  # extra env vars
    skip_permissions: bool = True  # --dangerously-skip-permissions
    system_prompt: str | None = None  # --prompt
    agent: str | None = None  # --agent
