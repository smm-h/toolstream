"""Pluggable history strategies for managing conversation message lists."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HistoryStrategy(Protocol):
    """Protocol for managing conversation history in DirectClient."""

    def append(self, message: dict) -> None:
        """Append a message to the history."""
        ...

    def messages_for_api(self) -> list[dict]:
        """Return messages to send to the API."""
        ...

    def on_usage(self, usage: dict) -> None:
        """Update internal state from an API usage response."""
        ...


class UnboundedHistory:
    """Default strategy: keeps all messages, no trimming.

    Preserves the original DirectClient behavior exactly.
    """

    def __init__(self) -> None:
        self._messages: list[dict] = []

    def append(self, message: dict) -> None:
        self._messages.append(message)

    def messages_for_api(self) -> list[dict]:
        return self._messages

    def on_usage(self, usage: dict) -> None:
        pass


class TokenBudgetHistory:
    """Strategy that drops oldest non-system messages when approaching the token budget.

    When total_tokens exceeds 90% of max_tokens, messages_for_api() trims
    the oldest non-system messages (preserving index 0, which is the system
    prompt) proportionally to bring the estimated usage under budget.
    """

    def __init__(self, max_tokens: int) -> None:
        self._max_tokens = max_tokens
        self._messages: list[dict] = []
        self._total_tokens: int = 0

    def append(self, message: dict) -> None:
        self._messages.append(message)

    def on_usage(self, usage: dict) -> None:
        self._total_tokens = usage.get("total_tokens", 0) or usage.get("prompt_tokens", 0)

    def messages_for_api(self) -> list[dict]:
        if self._total_tokens <= self._max_tokens * 0.9:
            return self._messages

        # Over budget: drop oldest non-system messages
        if len(self._messages) <= 1:
            return self._messages

        # Keep the system message (index 0), trim from the front of the rest
        system = self._messages[0]
        rest = self._messages[1:]

        # Drop messages proportional to how far over budget we are
        overshoot = self._total_tokens / self._max_tokens
        # Drop enough to get back to ~80% of budget
        drop_fraction = 1.0 - (0.8 / overshoot)
        drop_count = max(1, int(len(rest) * drop_fraction))

        trimmed = rest[drop_count:]
        self._messages = [system] + trimmed
        return self._messages
