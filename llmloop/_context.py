"""PipelineContext -- root context object for orchestrator sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineContext:
    """Root context for orchestrator sessions.

    Holds all injectable dependencies as named attributes.
    Tools resolve inject params via getattr on this object.
    """

    spawn_ctx: Any = None
    browser_ctx: Any = None
