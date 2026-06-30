"""Direct LLM API client via AI Gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from . import _builtin_tools
from ._history import HistoryStrategy, UnboundedHistory
from ._tools import Tool, collect_tools
from .config import SessionConfig
from .events import Error, StepFinish, StepStart, Text, ToolUse

logger = logging.getLogger(__name__)


def _safe_default(obj: object) -> str:
    """JSON serialization fallback: log a warning and return repr().

    Used as the ``default`` argument to ``json.dumps()`` when serializing
    tool results that may contain non-JSON-serializable objects (e.g.,
    dataclasses, bytes, custom response objects).
    """
    logger.warning(
        "tool_result_not_serializable: type=%s", type(obj).__name__,
    )
    return repr(obj)


_MAX_RETRIES = 3
_RETRYABLE_STATUS_CODES = frozenset({429, 503})


def _strip_provider(model: str) -> str:
    """Strip provider prefix from model name.

    'azure-cognitive-services/gpt-5.4' -> 'gpt-5.4'
    'gpt-5.4' -> 'gpt-5.4'
    """
    if "/" in model:
        return model.rsplit("/", 1)[1]
    return model


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _build_tool_definitions(tools: dict[str, Tool]) -> list[dict]:
    """Build OpenAI-format tool definitions from Tool objects."""
    defs = []
    for t in tools.values():
        defs.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        })
    return defs


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter, capped at 16s."""
    return min(2 ** attempt, 16) + random.uniform(0, 1)


async def _request_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
) -> dict:
    """POST with retries for transient errors and rate limits."""
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.post(url, json=json, headers=headers)
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                delay = _backoff(attempt)
                logger.warning(
                    "Request failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES + 1, e, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise
        except httpx.HTTPStatusError as e:
            if e.response.status_code in _RETRYABLE_STATUS_CODES:
                last_error = e
                if attempt < _MAX_RETRIES:
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after is not None:
                        try:
                            delay = min(float(retry_after), 60.0)
                        except ValueError:
                            delay = _backoff(attempt)
                    else:
                        delay = _backoff(attempt)
                    logger.warning(
                        "Request failed with %d (attempt %d/%d). Retrying in %.1fs",
                        e.response.status_code, attempt + 1, _MAX_RETRIES + 1, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            raise
    # Should never reach here, but satisfy type checker
    raise last_error  # type: ignore[misc]


class DirectClient:
    """Direct LLM API client (via AI Gateway) with tool calling."""

    def __init__(
        self,
        config: SessionConfig,
        *,
        tools: list[Tool] | None = None,
        tool_context: object | None = None,
        max_completion_tokens: int = 16384,
        http_client: httpx.AsyncClient | None = None,
        history: HistoryStrategy | None = None,
    ) -> None:
        if not config.api_key:
            raise ValueError("api_key is required for direct backend")
        if not config.base_url:
            raise ValueError("base_url is required for direct backend")
        if config.api_key and config.auth_style is None:
            raise ValueError(
                "auth_style is required when api_key is set. "
                "Use 'bearer' for OpenAI or 'x-api-key' for gateway."
            )
        if config.auth_style is not None and config.auth_style not in ("bearer", "x-api-key"):
            raise ValueError(
                f"auth_style must be 'bearer' or 'x-api-key', got {config.auth_style!r}"
            )

        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._api_key = config.api_key
        self._model = _strip_provider(config.model)
        if history is not None:
            self._history: HistoryStrategy = history
        elif config.history_strategy is not None:
            self._history: HistoryStrategy = config.history_strategy  # type: ignore[no-redef]
        else:
            self._history: HistoryStrategy = UnboundedHistory()  # type: ignore[no-redef]
        self._session_id = str(uuid.uuid4())
        self._cwd = config.cwd or os.getcwd()
        self._max_completion_tokens = max_completion_tokens
        self._tool_context = tool_context
        self._owns_client = http_client is None

        # Guard fields
        self._max_tool_rounds = config.max_tool_rounds
        self._send_timeout = config.send_timeout
        self._tool_call_timeout = config.tool_call_timeout
        self._max_turn_tokens = config.max_turn_tokens

        # Builtin context for inject resolution
        self._builtin_context: dict[str, Any] = {
            "cwd": self._cwd,
            "env": config.tool_env,
        }

        # HTTP client: reuse injected or create one
        if http_client is not None:
            self._client = http_client
        else:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
            )

        # Build tool registry: start with built-ins, override with user tools
        builtin_tools = collect_tools(_builtin_tools)
        self._builtin_names: set[str] = {t.name for t in builtin_tools}
        self._tools: dict[str, Tool] = {t.name: t for t in builtin_tools}

        if tools is not None:
            for t in tools:
                self._builtin_names.discard(t.name)
                self._tools[t.name] = t

        # Pre-compute API tool definitions
        self._tool_definitions = _build_tool_definitions(self._tools)

        # Build system prompt
        self._history.append({"role": "system", "content": config.system_prompt})

    @property
    def session_id(self) -> str:
        return self._session_id

    def _step_finish(self, msg_id: str, usage: dict, has_tool_calls: bool) -> StepFinish:
        """Build a StepFinish with per-step token values from the API response."""
        return StepFinish(
            session_id=self._session_id,
            message_id=msg_id,
            reason="tool-calls" if has_tool_calls else "stop",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
            cache_read_tokens=usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            cache_write_tokens=0,
            cost=0.0,
            timestamp=_timestamp_ms(),
        )

    async def send(self, message: str) -> AsyncIterator[StepStart | Text | ToolUse | StepFinish | Error]:
        """Send a message and yield events. Handles the tool-calling loop internally."""
        self._history.append({"role": "user", "content": message})
        msg_id = str(uuid.uuid4())

        yield StepStart(
            session_id=self._session_id,
            message_id=msg_id,
            timestamp=_timestamp_ms(),
        )

        iteration = 0
        start = time.monotonic()
        cumulative_tokens = 0

        while True:
            iteration += 1
            if self._max_tool_rounds is not None and iteration > self._max_tool_rounds:
                yield Error(
                    session_id=self._session_id,
                    name="max_iterations_exceeded",
                    message=f"Stopped after {self._max_tool_rounds} tool rounds",
                    data={"max_tool_rounds": self._max_tool_rounds, "iteration": iteration},
                    timestamp=_timestamp_ms(),
                )
                break
            if self._send_timeout is not None and time.monotonic() - start > self._send_timeout:
                yield Error(
                    session_id=self._session_id,
                    name="send_timeout_exceeded",
                    message=f"Stopped after {self._send_timeout}s",
                    data={"send_timeout": self._send_timeout, "elapsed": time.monotonic() - start},
                    timestamp=_timestamp_ms(),
                )
                break

            response = await self._chat_completion(self._history.messages_for_api())
            usage = response.get("usage", {})

            choice = response["choices"][0]
            assistant_msg = choice["message"]

            # Detect output truncation (finish_reason="length")
            finish_reason = choice.get("finish_reason", "")
            if finish_reason == "length":
                logger.warning(
                    "Model output truncated at %d tokens (finish_reason=length)",
                    self._max_completion_tokens,
                )
                yield Error(
                    session_id=self._session_id,
                    name="output_truncated",
                    message=(
                        f"Model output truncated at {self._max_completion_tokens} tokens. "
                        f"Increase max_completion_tokens or simplify the task."
                    ),
                    data={
                        "finish_reason": finish_reason,
                        "max_completion_tokens": self._max_completion_tokens,
                    },
                    timestamp=_timestamp_ms(),
                )
                break

            # Build the message dict to append to conversation.
            # OpenAI requires content (even if null) when tool_calls are present.
            msg_to_append: dict = {
                "role": "assistant",
                "content": assistant_msg.get("content"),
            }
            if assistant_msg.get("tool_calls"):
                msg_to_append["tool_calls"] = assistant_msg["tool_calls"]
            self._history.append(msg_to_append)

            # Yield text if present
            if assistant_msg.get("content"):
                yield Text(
                    session_id=self._session_id,
                    message_id=msg_id,
                    text=assistant_msg["content"],
                    timestamp=_timestamp_ms(),
                )

            # Check for tool calls (may be None or missing)
            tool_calls = assistant_msg.get("tool_calls") or []

            # Execute tool calls and add results
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                result = await self._dispatch_tool(func_name, func_args)

                yield ToolUse(
                    session_id=self._session_id,
                    message_id=msg_id,
                    tool=func_name,
                    call_id=tc["id"],
                    status="completed",
                    input=func_args,
                    output=result,
                    title=func_name,
                    timestamp=_timestamp_ms(),
                )

                self._history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result if isinstance(result, str) else json.dumps(result, default=_safe_default),
                })

            self._history.on_usage(usage)
            step_input = usage.get("prompt_tokens", 0)
            step_output = usage.get("completion_tokens", 0)
            cumulative_tokens += step_input + step_output
            yield self._step_finish(msg_id, usage, has_tool_calls=bool(tool_calls))
            if not tool_calls:
                break
            if self._max_turn_tokens is not None and cumulative_tokens > self._max_turn_tokens:
                yield Error(
                    session_id=self._session_id,
                    name="token_budget_exceeded",
                    message=f"Stopped after {cumulative_tokens} tokens (budget: {self._max_turn_tokens})",
                    data={"cumulative_tokens": cumulative_tokens, "max_turn_tokens": self._max_turn_tokens},
                    timestamp=_timestamp_ms(),
                )
                break

    async def _chat_completion(self, messages: list[dict]) -> dict:
        """Call LLM chat completions via AI Gateway with retry."""
        url = self._base_url
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._config.auth_style == "bearer":
            headers["Authorization"] = f"Bearer {self._api_key}"
        elif self._config.auth_style == "x-api-key":
            headers["x-api-key"] = self._api_key
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": self._tool_definitions,
            "max_completion_tokens": self._max_completion_tokens,
        }
        if self._config.metadata:
            body["metadata"] = self._config.metadata

        return await _request_with_retry(
            self._client, url, json=body, headers=headers,
        )

    async def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        """Dispatch a tool call by name. Returns the result string."""
        tool_obj = self._tools.get(name)
        if tool_obj is None:
            return f"Error: unknown tool '{name}'"

        if name in self._builtin_names:
            # Resolve inject params from builtin context
            kwargs = {p: self._builtin_context[p] for p in tool_obj.inject}
            try:
                result = await tool_obj.handler(**args, **kwargs)
            except Exception as e:
                return f"Error: {e}"
        else:
            # User tools: inject context params, then call handler
            for param_name in tool_obj.inject:
                if self._tool_context is None:
                    raise RuntimeError(
                        f"Tool '{name}' requires tool_context "
                        f"(inject=['{param_name}']) but tool_context is None"
                    )
                try:
                    args[param_name] = getattr(self._tool_context, param_name)
                except AttributeError:
                    raise AttributeError(
                        f"tool_context ({type(self._tool_context).__name__}) "
                        f"has no attribute '{param_name}' "
                        f"required by tool '{name}'"
                    )
            try:
                result = await tool_obj.handler(**args)
            except Exception as e:
                return f"Error: {e}"

        return result

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> DirectClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
