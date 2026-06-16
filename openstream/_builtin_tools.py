"""Standalone implementations of the built-in tools for direct-mode LLM sessions.

Each function is an async def decorated with @tool that receives `cwd` via
injection (excluded from the JSON Schema sent to the LLM, passed explicitly
by the DirectClient's dispatch loop).
"""

from __future__ import annotations

import asyncio
import os
import shlex
from glob import glob as _stdlib_glob
from pathlib import Path

from ._tools import tool

_MAX_OUTPUT_CHARS = 50_000

ENV_BLOCKLIST = frozenset({
    "SUPABASE_DB_PASSWORD",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SHOPKEEP_DATABASE_URL",
    "SHOPKEEP_AI_GATEWAY_API_KEY",
    "AI_GATEWAY_API_KEY",
    "SHOPKEEP_ANTHROPIC_API_KEY",
})


def _resolve_path(file_path: str, cwd: str) -> Path:
    """Resolve *file_path*, making relative paths absolute against *cwd*."""
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(cwd) / path
    return path


@tool("builtin", inject=["cwd"])
async def read(file_path: str, cwd: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file and return its contents with line numbers."""
    path = _resolve_path(file_path, cwd)
    lines = path.read_text().splitlines()
    selected = lines[offset : offset + limit]
    return "\n".join(f"{i + offset + 1}: {line}" for i, line in enumerate(selected))


@tool("builtin", inject=["cwd"])
async def write(file_path: str, content: str, cwd: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    path = _resolve_path(file_path, cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return f"Wrote {len(content)} bytes to {file_path}"


@tool("builtin", inject=["cwd"])
async def bash(command: str, cwd: str, timeout: int = 120) -> str:
    """Run a shell command and return stdout+stderr combined."""
    filtered_env = {k: v for k, v in os.environ.items() if k not in ENV_BLOCKLIST}
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        env=filtered_env,
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


@tool("builtin", inject=["cwd"])
async def edit(file_path: str, old_string: str, new_string: str, cwd: str) -> str:
    """Edit a file by replacing the first occurrence of old_string with new_string."""
    path = _resolve_path(file_path, cwd)
    content = path.read_text()
    if old_string not in content:
        return f"Error: old_string not found in {file_path}"
    content = content.replace(old_string, new_string, 1)
    path.write_text(content)
    return f"Edited {file_path}"


@tool("builtin", inject=["cwd"])
async def grep(pattern: str, path: str, cwd: str, include: str | None = None) -> str:
    """Search for a pattern in files using grep -rn, piped to head -50."""
    cmd = f"grep -rn {shlex.quote(pattern)} {shlex.quote(path)}"
    if include:
        cmd += f" --include={shlex.quote(include)}"
    return await bash(cmd + " | head -50", cwd=cwd, timeout=30)


@tool("builtin", name="glob", inject=["cwd"])
async def glob_files(pattern: str, cwd: str) -> str:
    """Find files matching a glob pattern. Returns up to 100 matches."""
    matches = _stdlib_glob(pattern, recursive=True)
    return "\n".join(matches[:100])
