"""ToolContext -- base class for tool dependency injection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolContext:
    """Base class for tool dependency injection.

    Subclass with your application's injectable attributes.
    Tools declare inject parameters that are resolved via getattr on this object.
    """
