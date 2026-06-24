"""Direct LLM API client via AI Gateway."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from . import _builtin_tools
from ._tools import Tool, collect_tools
from .config import SessionConfig
from .events import Error, StepFinish, StepStart, Text, ToolUse

_MAX_RETRIES = 1


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
        self._messages: list[dict] = []
        self._session_id = str(uuid.uuid4())
        self._cwd = config.cwd or os.getcwd()
        self._max_completion_tokens = max_completion_tokens
        self._tool_context = tool_context
        self._owns_client = http_client is None

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
        self._messages.append({"role": "system", "content": config.system_prompt})

    @property
    def session_id(self) -> str:
        return self._session_id

    async def send(self, message: str) -> AsyncIterator[StepStart | Text | ToolUse | StepFinish | Error]:
        """Send a message and yield events. Handles the tool-calling loop internally."""
        self._messages.append({"role": "user", "content": message})
        msg_id = str(uuid.uuid4())

        yield StepStart(
            session_id=self._session_id,
            message_id=msg_id,
            timestamp=_timestamp_ms(),
        )

        total_input_tokens = 0
        total_output_tokens = 0

        while True:
            response = await self._chat_completion(self._messages)
            usage = response.get("usage", {})
            total_input_tokens += usage.get("prompt_tokens", 0)
            total_output_tokens += usage.get("completion_tokens", 0)

            choice = response["choices"][0]
            assistant_msg = choice["message"]

            # Build the message dict to append to conversation.
            # OpenAI requires content (even if null) when tool_calls are present.
            msg_to_append: dict = {
                "role": "assistant",
                "content": assistant_msg.get("content"),
            }
            if assistant_msg.get("tool_calls"):
                msg_to_append["tool_calls"] = assistant_msg["tool_calls"]
            self._messages.append(msg_to_append)

            # Yield text if present
            if assistant_msg.get("content"):
                yield Text(
                    session_id=self._session_id,
                    message_id=msg_id,
                    text=assistant_msg["content"],
                    timestamp=_timestamp_ms(),
                )

            # Check for tool calls
            tool_calls = assistant_msg.get("tool_calls", [])
            if not tool_calls:
                # No tool calls -- conversation turn is done
                yield StepFinish(
                    session_id=self._session_id,
                    message_id=msg_id,
                    reason="stop",
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
                    cache_read_tokens=usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                    cache_write_tokens=0,
                    cost=0.0,
                    timestamp=_timestamp_ms(),
                )
                break

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

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result if isinstance(result, str) else json.dumps(result),
                })

            # Yield StepFinish for this tool-calling round
            yield StepFinish(
                session_id=self._session_id,
                message_id=msg_id,
                reason="tool-calls",
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                reasoning_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost=0.0,
                timestamp=_timestamp_ms(),
            )

    async def _chat_completion(self, messages: list[dict]) -> dict:
        """Call LLM chat completions via AI Gateway with timeout and retry."""
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

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.post(url, json=body, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.ReadTimeout as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    continue
                raise
            except httpx.HTTPStatusError:
                raise

        # Should never reach here, but satisfy type checker
        raise last_error  # type: ignore[misc]

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
