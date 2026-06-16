from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ._direct import DirectClient
from .config import SessionConfig
from .events import Result, StepFinish


class AsyncSession:
    """Async session wrapping the direct LLM API client."""

    def __init__(self, config: SessionConfig):
        self._config = config
        self._direct = DirectClient(
            config,
            tools=config.tools,
            tool_context=config.tool_context,
            max_completion_tokens=config.max_completion_tokens,
        )
        self._turn_count = 0
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    async def __aenter__(self) -> AsyncSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def send(self, message: str) -> AsyncIterator[Any]:
        """Send a message and yield events. Yields a Result summary at the end."""
        async for event in self._send_direct(message):
            yield event

    async def _send_direct(self, message: str) -> AsyncIterator[Any]:
        """Send via direct API client."""
        self._turn_count += 1

        turn_input_tokens = 0
        turn_output_tokens = 0
        turn_reasoning_tokens = 0
        turn_cache_read_tokens = 0
        turn_cache_write_tokens = 0
        turn_cost = 0.0
        steps = 0

        async for event in self._direct.send(message):
            if isinstance(event, StepFinish):
                turn_input_tokens += event.input_tokens
                turn_output_tokens += event.output_tokens
                turn_reasoning_tokens += event.reasoning_tokens
                turn_cache_read_tokens += event.cache_read_tokens
                turn_cache_write_tokens += event.cache_write_tokens
                turn_cost += event.cost
                steps += 1
            yield event

        # Update totals
        self._total_input_tokens += turn_input_tokens
        self._total_output_tokens += turn_output_tokens
        self._total_cost += turn_cost

        # Yield result summary
        yield Result(
            session_id=self._direct.session_id,
            total_input_tokens=turn_input_tokens,
            total_output_tokens=turn_output_tokens,
            total_cost=turn_cost,
            steps=steps,
        )

    async def close(self) -> None:
        """Close the session and clean up."""
        await self._direct.close()

    @property
    def session_id(self) -> str | None:
        return self._direct.session_id

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def turn_count(self) -> int:
        return self._turn_count


class SyncSession:
    """Sync wrapper around AsyncSession using a dedicated event loop thread."""

    def __init__(self, config: SessionConfig):
        self._config = config
        self._async_session: AsyncSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def _start_loop(self) -> None:
        """Run the event loop in a background thread."""
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def __enter__(self) -> SyncSession:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()
        self._async_session = AsyncSession(self._config)
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _run_coroutine(self, coro: Any) -> Any:
        """Run a coroutine on the background event loop and return the result."""
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def send(self, message: str) -> Iterator[Any]:
        """Send a message and yield events incrementally via a queue bridge."""
        assert self._async_session is not None
        assert self._loop is not None

        q: queue.Queue = queue.Queue()
        sentinel = object()

        async def _produce() -> None:
            try:
                async for event in self._async_session.send(message):  # type: ignore[union-attr]
                    q.put(event)
            except Exception as e:
                q.put(e)
            finally:
                q.put(sentinel)

        asyncio.run_coroutine_threadsafe(_produce(), self._loop)

        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self) -> None:
        """Close the session and shut down the event loop."""
        if self._async_session is not None:
            self._run_coroutine(self._async_session.close())
            self._async_session = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            self._loop.close()
            self._loop = None
            self._thread = None

    @property
    def session_id(self) -> str | None:
        return self._async_session.session_id if self._async_session else None

    @property
    def total_cost(self) -> float:
        return self._async_session.total_cost if self._async_session else 0.0

    @property
    def turn_count(self) -> int:
        return self._async_session.turn_count if self._async_session else 0
