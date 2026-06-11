"""Direct LLM API client via AI Gateway -- replaces the opencode subprocess."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import time
import uuid
from collections.abc import AsyncIterator
from glob import glob
from pathlib import Path

import httpx

from .config import SessionConfig
from .events import Error, StepFinish, StepStart, Text, ToolUse

_DEFAULT_SYSTEM_PROMPT = (
    "You are a coding assistant. You have access to tools for reading files, "
    "writing files, editing files, running bash commands, searching with grep, "
    "and finding files with glob. Use these tools to accomplish the user's task. "
    "Work in the directory: {cwd}"
)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file and return its contents with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to read"},
                    "offset": {
                        "type": "integer",
                        "description": "Line offset to start reading from (0-based)",
                        "default": 0,
                    },
                    "limit": {"type": "integer", "description": "Maximum number of lines to read", "default": 2000},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file, creating parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a file by replacing the first occurrence of old_string with new_string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to edit"},
                    "old_string": {"type": "string", "description": "The exact string to find and replace"},
                    "new_string": {"type": "string", "description": "The replacement string"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern in files using grep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search in", "default": "."},
                    "include": {
                        "type": "string",
                        "description": "File glob pattern to include (e.g. '*.py')",
                        "default": "",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (supports ** for recursive)"},
                },
                "required": ["pattern"],
            },
        },
    },
]

_MAX_OUTPUT_CHARS = 50_000
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


class DirectClient:
    """Direct LLM API client (via AI Gateway) with tool calling."""

    def __init__(self, config: SessionConfig) -> None:
        if not config.api_key:
            raise ValueError("api_key is required for direct backend")
        if not config.base_url:
            raise ValueError("base_url is required for direct backend")

        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._api_key = config.api_key
        self._model = _strip_provider(config.model)
        self._messages: list[dict] = []
        self._tools = TOOL_DEFINITIONS
        self._session_id = str(uuid.uuid4())
        self._cwd = config.cwd or os.getcwd()

        # Build system prompt
        system_prompt = config.system_prompt or _DEFAULT_SYSTEM_PROMPT.format(cwd=self._cwd)
        self._messages.append({"role": "system", "content": system_prompt})

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

            # Build the message dict to append to conversation
            msg_to_append: dict = {"role": "assistant"}
            if assistant_msg.get("content"):
                msg_to_append["content"] = assistant_msg["content"]
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

                result = await self._execute_tool(func_name, func_args)

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
                    "content": result,
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
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "messages": messages,
            "tools": self._tools,
            "max_completion_tokens": 16384,
            "metadata": {"service": "shopkeep"},
        }

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
                ) as client:
                    response = await client.post(url, json=body, headers=headers)
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

    async def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool by name and return the result string."""
        try:
            if name == "read":
                return await self._tool_read(
                    args["file_path"],
                    offset=args.get("offset", 0),
                    limit=args.get("limit", 2000),
                )
            if name == "write":
                return await self._tool_write(args["file_path"], args["content"])
            if name == "bash":
                return await self._tool_bash(
                    args["command"],
                    timeout=args.get("timeout", 120),
                )
            if name == "edit":
                return await self._tool_edit(
                    args["file_path"], args["old_string"], args["new_string"],
                )
            if name == "grep":
                return await self._tool_grep(
                    args["pattern"],
                    path=args.get("path", "."),
                    include=args.get("include", ""),
                )
            if name == "glob":
                return await self._tool_glob(args["pattern"])
            return f"Error: unknown tool '{name}'"
        except Exception as e:
            return f"Error: {e}"

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a path, making relative paths absolute against cwd."""
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(self._cwd) / path
        return path

    async def _tool_read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        path = self._resolve_path(file_path)
        lines = path.read_text().splitlines()
        selected = lines[offset : offset + limit]
        return "\n".join(f"{i + offset + 1}: {line}" for i, line in enumerate(selected))

    async def _tool_write(self, file_path: str, content: str) -> str:
        path = self._resolve_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Wrote {len(content)} bytes to {file_path}"

    async def _tool_bash(self, command: str, timeout: int = 120) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self._cwd,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode(errors="replace")
            if len(output) > _MAX_OUTPUT_CHARS:
                output = output[:_MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(output)} total chars)"
            return output
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Command timed out after {timeout}s"

    async def _tool_edit(self, file_path: str, old_string: str, new_string: str) -> str:
        path = self._resolve_path(file_path)
        content = path.read_text()
        if old_string not in content:
            return f"Error: old_string not found in {file_path}"
        content = content.replace(old_string, new_string, 1)
        path.write_text(content)
        return f"Edited {file_path}"

    async def _tool_grep(self, pattern: str, path: str = ".", include: str = "") -> str:
        cmd = f"grep -rn {shlex.quote(pattern)} {shlex.quote(path)}"
        if include:
            cmd += f" --include={shlex.quote(include)}"
        return await self._tool_bash(cmd + " | head -50", timeout=30)

    async def _tool_glob(self, pattern: str) -> str:
        matches = glob(pattern, recursive=True)
        return "\n".join(matches[:100])

    async def close(self) -> None:
        """No subprocess to clean up."""
