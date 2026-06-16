"""Agent definition loading and discovery."""

from __future__ import annotations

import importlib.resources
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "ToolRef",
    "AgentSandbox",
    "AgentDefinition",
    "load_agent",
    "discover_agents",
    "resolve_prompt",
]


@dataclass(frozen=True)
class ToolRef:
    name: str


@dataclass(frozen=True)
class AgentSandbox:
    tools: list[str] | None = None
    skip_permissions: bool = False


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    prompt_template: str
    version: str
    description: str = ""
    tools: list[ToolRef] | None = None
    sandbox: AgentSandbox | None = None
    model: str | None = None


def _parse_agent_data(data: dict, source: str) -> AgentDefinition:
    """Parse a dict (from JSON) into an AgentDefinition.

    Args:
        data: Parsed JSON dict.
        source: Human-readable origin for error messages (file path, package).
    """
    tools: list[ToolRef] | None = None
    if "tools" in data:
        tools = [ToolRef(name=entry["name"]) for entry in (data["tools"] or [])]

    sandbox: AgentSandbox | None = None
    if "sandbox" in data:
        sandbox_data = data["sandbox"]
        sandbox = AgentSandbox(
            tools=sandbox_data.get("tools"),
            skip_permissions=sandbox_data.get("skip_permissions", False),
        )

    return AgentDefinition(
        name=data["name"],
        prompt_template=data["prompt_template"],
        version=data["version"],
        description=data.get("description", ""),
        tools=tools,
        sandbox=sandbox,
        model=data.get("model"),
    )


def load_agent(path: str | Path) -> AgentDefinition:
    """Load an agent definition from a JSON file.

    Args:
        path: Path to a ``.agent.json`` file. Bare names (no path separator,
              no ``.json`` suffix) are rejected -- use :func:`discover_agents`
              for name-based lookup.

    Raises:
        FileNotFoundError: If *path* is a bare name or the file does not exist.
    """
    if isinstance(path, str) and "/" not in path and os.sep not in path and not path.endswith(".json"):
        raise FileNotFoundError(
            f"Agent '{path}' not found (bare name resolution requires discover_agents)"
        )

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return _parse_agent_data(data, source=str(p))


def _try_load(
    file_path: Path,
    seen: dict[str, AgentDefinition],
) -> None:
    """Attempt to load a single agent file, deduplicating by name."""
    try:
        agent = load_agent(file_path)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Skipping %s: %s", file_path, exc)
        return
    except FileNotFoundError as exc:
        logger.warning("Skipping %s: %s", file_path, exc)
        return

    if agent.name in seen:
        logger.warning(
            "Duplicate agent '%s' in %s (already loaded); skipping",
            agent.name,
            file_path,
        )
        return

    seen[agent.name] = agent


def discover_agents(
    cwd: str | Path | None = None,
    paths: list[str | Path] | None = None,
    packages: list[str] | None = None,
) -> list[AgentDefinition]:
    """Discover agent definitions from directories and packages.

    Search order (first occurrence of a name wins):

    1. ``<cwd>/.openstream/agents/*.agent.json``
    2. Each directory in *paths*: ``<dir>/*.agent.json``
    3. Each Python package in *packages*: resources matching ``*.agent.json``

    Returns:
        Agent definitions sorted alphabetically by name.
    """
    seen: dict[str, AgentDefinition] = {}

    # 1. cwd-local agents
    if cwd is not None:
        agents_dir = Path(cwd) / ".openstream" / "agents"
        if agents_dir.is_dir():
            for f in sorted(agents_dir.glob("*.agent.json")):
                _try_load(f, seen)

    # 2. Explicit directory paths
    if paths is not None:
        for dir_path in paths:
            d = Path(dir_path)
            if not d.is_dir():
                logger.warning("Agent path %s is not a directory; skipping", d)
                continue
            for f in sorted(d.glob("*.agent.json")):
                _try_load(f, seen)

    # 3. Python packages
    if packages is not None:
        for package_name in packages:
            try:
                pkg_files = importlib.resources.files(package_name)
            except ModuleNotFoundError:
                logger.warning("Package '%s' not found; skipping", package_name)
                continue

            for item in pkg_files.iterdir():
                if not item.name.endswith(".agent.json"):
                    continue
                try:
                    data = json.loads(item.read_text(encoding="utf-8"))
                    agent = _parse_agent_data(data, source=f"{package_name}/{item.name}")
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    logger.warning(
                        "Skipping %s/%s: %s", package_name, item.name, exc
                    )
                    continue

                if agent.name in seen:
                    logger.warning(
                        "Duplicate agent '%s' in %s/%s (already loaded); skipping",
                        agent.name,
                        package_name,
                        item.name,
                    )
                    continue

                seen[agent.name] = agent

    return sorted(seen.values(), key=lambda a: a.name)


def resolve_prompt(template: str, variables: dict[str, str]) -> str:
    """Substitute ``{key}`` placeholders in a prompt template.

    Uses :meth:`str.replace` (not :meth:`str.format`) so only the
    supplied *variables* are touched; literal braces in the template
    are preserved.

    Raises:
        ValueError: If any original placeholder has no matching variable.
    """
    # Find all original placeholders before substitution.
    original_placeholders = set(re.findall(r"\{(\w+)\}", template))

    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", value)

    # Only check placeholders that were in the original template.
    unresolved = [p for p in sorted(original_placeholders) if p not in variables]
    if unresolved:
        raise ValueError(f"Unresolved template variables: {', '.join(unresolved)}")

    return result
