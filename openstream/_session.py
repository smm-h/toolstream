from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ._process import OpenCodeProcess
from ._protocol import Event, parse_event
from .config import SessionConfig
from .events import Error, Result, StepFinish


class AsyncSession:
    """Async session wrapping opencode."""

    def __init__(self, config: SessionConfig):
        self._config = config
        self._process = OpenCodeProcess(config)
        self._turn_count = 0
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    async def __aenter__(self) -> AsyncSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def send(self, message: str) -> AsyncIterator[Event | Result]:
        """Send a message and yield events. Yields a Result summary at the end."""
        if self._turn_count == 0:
            await self._process.start(message)
        else:
            await self._process.continue_with(message)

        self._turn_count += 1

        turn_input_tokens = 0
        turn_output_tokens = 0
        turn_cost = 0.0
        steps = 0
        session_id_captured = False

        while True:
            line = await self._process.readline()
            if line is None:
                break

            event = parse_event(line)
            if event is None:
                continue

            # Capture session ID from the first event
            if not session_id_captured and hasattr(event, "session_id") and event.session_id:
                self._process.session_id = event.session_id
                session_id_captured = True

            if isinstance(event, StepFinish):
                turn_input_tokens += event.input_tokens
                turn_output_tokens += event.output_tokens
                turn_cost += event.cost
                steps += 1

            yield event

        # Wait for process to finish
        exit_code = await self._process.wait()
        if exit_code != 0 and exit_code != -1:
            stderr = await self._process.read_stderr()
            yield Error(
                session_id=self._process.session_id or "",
                name="process_error",
                message=f"opencode exited with code {exit_code}: {stderr.strip()}",
                data={"exit_code": exit_code},
                timestamp=0,
            )

        # Update totals
        self._total_input_tokens += turn_input_tokens
        self._total_output_tokens += turn_output_tokens
        self._total_cost += turn_cost

        # Yield result summary
        yield Result(
            session_id=self._process.session_id or "",
            total_input_tokens=turn_input_tokens,
            total_output_tokens=turn_output_tokens,
            total_cost=turn_cost,
            steps=steps,
        )

    async def close(self) -> None:
        """Close the session and clean up."""
        await self._process.close()

    @property
    def session_id(self) -> str | None:
        return self._process.session_id

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

    def send(self, message: str) -> Iterator[Event | Result]:
        """Send a message and yield events."""
        assert self._async_session is not None
        assert self._loop is not None

        # Collect events from the async iterator via the background loop
        async def _collect() -> list[Event | Result]:
            events: list[Event | Result] = []
            async for event in self._async_session.send(message):  # type: ignore[union-attr]
                events.append(event)
            return events

        events = self._run_coroutine(_collect())
        yield from events

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
