from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from .config import SessionConfig


def _find_opencode_binary(config: SessionConfig) -> str:
    """Locate the opencode binary."""
    if config.opencode_binary:
        if not Path(config.opencode_binary).exists():
            raise RuntimeError(f"opencode binary not found at: {config.opencode_binary}")
        return config.opencode_binary

    # Check PATH
    found = shutil.which("opencode")
    if found:
        return found

    # Check default install location
    default = Path.home() / ".opencode" / "bin" / "opencode"
    if default.exists():
        return str(default)

    raise RuntimeError(
        "opencode binary not found. Install opencode or set opencode_binary in SessionConfig."
    )


class OpenCodeProcess:
    """Manages an opencode subprocess."""

    def __init__(self, config: SessionConfig):
        self._config = config
        self._binary = _find_opencode_binary(config)
        self._process: asyncio.subprocess.Process | None = None
        self._session_id: str | None = None

    def _build_command(self, message: str, *, continue_session: bool = False) -> list[str]:
        """Build the opencode command line."""
        cmd = [self._binary, "run", message, "--format", "json", "-m", self._config.model]

        if self._config.skip_permissions:
            cmd.append("--dangerously-skip-permissions")

        if self._config.system_prompt:
            cmd.extend(["--prompt", self._config.system_prompt])

        if self._config.agent:
            cmd.extend(["--agent", self._config.agent])

        if continue_session and self._session_id:
            cmd.extend(["--session", self._session_id, "--continue"])

        return cmd

    def _build_env(self) -> dict[str, str]:
        """Build environment variables for the subprocess."""
        env = dict(os.environ)
        env.update(self._config.env)
        return env

    async def start(self, message: str) -> None:
        """Start the subprocess with an initial message."""
        cmd = self._build_command(message, continue_session=False)
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._config.cwd,
            env=self._build_env(),
        )

    async def continue_with(self, message: str) -> None:
        """Start a new subprocess continuing the existing session."""
        await self.close()
        cmd = self._build_command(message, continue_session=True)
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._config.cwd,
            env=self._build_env(),
        )

    async def readline(self) -> str | None:
        """Read one line from stdout. Returns None at EOF."""
        if self._process is None or self._process.stdout is None:
            return None
        line = await self._process.stdout.readline()
        if not line:
            return None
        return line.decode("utf-8", errors="replace")

    async def read_stderr(self) -> str:
        """Read all stderr output (for error reporting)."""
        if self._process is None or self._process.stderr is None:
            return ""
        data = await self._process.stderr.read()
        return data.decode("utf-8", errors="replace")

    async def wait(self) -> int:
        """Wait for the subprocess to exit. Returns exit code."""
        if self._process is None:
            return -1
        return await self._process.wait()

    async def close(self) -> None:
        """Kill the subprocess if running."""
        if self._process is not None:
            try:
                if self._process.returncode is None:
                    self._process.terminate()
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self._process.kill()
                        await self._process.wait()
            except ProcessLookupError:
                pass
            self._process = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value
