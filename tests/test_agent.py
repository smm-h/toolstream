"""Tests for agent definition loading, discovery, and prompt resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolstream._agent import (
    AgentDefinition,
    AgentSandbox,
    ToolRef,
    discover_agents,
    load_agent,
    resolve_prompt,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "agents"

# ============================================================
# Helpers
# ============================================================


def _make_agent_json(
    name="test-agent",
    version="1.0",
    prompt_template="Hello {name}",
    description="A test agent",
    tools=None,
    sandbox=None,
    model=None,
):
    data = {
        "name": name,
        "version": version,
        "prompt_template": prompt_template,
        "description": description,
    }
    if tools is not None:
        data["tools"] = tools
    if sandbox is not None:
        data["sandbox"] = sandbox
    if model is not None:
        data["model"] = model
    return data


def _write_agent_json(tmp_path, data, filename=None):
    if filename is None:
        filename = f"{data['name']}.agent.json"
    path = tmp_path / filename
    path.write_text(json.dumps(data))
    return path


# ============================================================
# TestLoadAgent -- loading .agent.json files
# ============================================================


class TestLoadAgent:
    def test_load_old_format(self, tmp_path):
        """Old-format tools (name + description + input_schema + server) load correctly;
        only the name is kept."""
        data = _make_agent_json(
            tools=[
                {
                    "name": "bash",
                    "description": "Execute a shell command.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                    "server": "test-tools",
                },
                {
                    "name": "navigate",
                    "description": "Navigate to a URL.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                        },
                        "required": ["url"],
                    },
                    "server": "test-crawler",
                },
            ]
        )
        path = _write_agent_json(tmp_path, data)
        agent = load_agent(path)

        assert agent.name == "test-agent"
        assert agent.tools is not None
        assert len(agent.tools) == 2
        assert agent.tools[0] == ToolRef(name="bash")
        assert agent.tools[1] == ToolRef(name="navigate")

    def test_load_new_format(self, tmp_path):
        """New-format tools (just name) load correctly."""
        data = _make_agent_json(
            tools=[
                {"name": "Read"},
                {"name": "Write"},
            ]
        )
        path = _write_agent_json(tmp_path, data)
        agent = load_agent(path)

        assert agent.tools is not None
        assert len(agent.tools) == 2
        assert agent.tools[0] == ToolRef(name="Read")
        assert agent.tools[1] == ToolRef(name="Write")

    def test_load_with_sandbox(self, tmp_path):
        """Sandbox with tools list and skip_permissions parses correctly."""
        data = _make_agent_json(
            sandbox={
                "tools": ["Read", "Write", "bash"],
                "skip_permissions": True,
            }
        )
        path = _write_agent_json(tmp_path, data)
        agent = load_agent(path)

        assert agent.sandbox is not None
        assert agent.sandbox.tools == ["Read", "Write", "bash"]
        assert agent.sandbox.skip_permissions is True

    def test_load_without_sandbox(self, tmp_path):
        """Missing sandbox key results in None."""
        data = _make_agent_json()
        path = _write_agent_json(tmp_path, data)
        agent = load_agent(path)

        assert agent.sandbox is None

    def test_load_with_model(self, tmp_path):
        """Top-level model field is preserved."""
        data = _make_agent_json(model="claude-sonnet")
        path = _write_agent_json(tmp_path, data)
        agent = load_agent(path)

        assert agent.model == "claude-sonnet"

    def test_load_without_model(self, tmp_path):
        """Missing model key results in None."""
        data = _make_agent_json()
        path = _write_agent_json(tmp_path, data)
        agent = load_agent(path)

        assert agent.model is None

    def test_load_file_not_found(self, tmp_path):
        """Nonexistent file raises FileNotFoundError."""
        path = tmp_path / "nonexistent.agent.json"
        with pytest.raises(FileNotFoundError):
            load_agent(path)

    def test_load_empty_tools_list(self, tmp_path):
        """An explicit empty tools list means 'no tools allowed' and must
        remain [] -- it must NOT be conflated with None (tools unspecified)."""
        data = _make_agent_json(tools=[])
        path = _write_agent_json(tmp_path, data)
        agent = load_agent(path)

        assert agent.tools == []

    def test_load_without_tools(self, tmp_path):
        """Missing tools key results in None (tools unspecified / use defaults),
        which is distinct from an empty list (no tools allowed)."""
        data = _make_agent_json()
        path = _write_agent_json(tmp_path, data)
        agent = load_agent(path)

        assert agent.tools is None

    def test_load_bare_name(self):
        """Bare name (no separators, no .json) raises FileNotFoundError
        with a message about discover_agents."""
        with pytest.raises(FileNotFoundError, match="bare name resolution"):
            load_agent("my-agent")


# ============================================================
# TestDiscoverAgents -- multi-source agent discovery
# ============================================================


class TestDiscoverAgents:
    def test_discover_from_directory(self, tmp_path):
        """Agents in an explicit paths directory are found and sorted by name."""
        _write_agent_json(tmp_path, _make_agent_json(name="beta"))
        _write_agent_json(tmp_path, _make_agent_json(name="alpha"))

        agents = discover_agents(paths=[tmp_path])

        assert len(agents) == 2
        assert agents[0].name == "alpha"
        assert agents[1].name == "beta"

    def test_discover_from_cwd(self, tmp_path):
        """Agents in <cwd>/.toolstream/agents/ are discovered."""
        agents_dir = tmp_path / ".toolstream" / "agents"
        agents_dir.mkdir(parents=True)
        _write_agent_json(agents_dir, _make_agent_json(name="local-agent"))

        agents = discover_agents(cwd=tmp_path)

        assert len(agents) == 1
        assert agents[0].name == "local-agent"

    def test_discover_from_fixtures(self):
        """Agent fixtures in tests/fixtures/agents/ are discovered."""
        agents = discover_agents(paths=[FIXTURES_DIR])

        names = {a.name for a in agents}
        assert "explorer" in names
        assert "orchestrator" in names
        assert len(agents) >= 2

    def test_discover_dedup(self, tmp_path):
        """First-seen wins: paths entry listed first overrides later entry."""
        # Create a local "explorer" with a different version to distinguish it.
        _write_agent_json(
            tmp_path,
            _make_agent_json(
                name="explorer",
                version="99.0",
                prompt_template="Local explorer {name}",
            ),
        )

        agents = discover_agents(paths=[tmp_path, FIXTURES_DIR])

        explorers = [a for a in agents if a.name == "explorer"]
        assert len(explorers) == 1
        assert explorers[0].version == "99.0"

    def test_discover_empty(self):
        """No sources returns empty list."""
        agents = discover_agents()
        assert agents == []

    def test_discover_nonexistent_package(self):
        """Nonexistent package returns empty list without crashing."""
        agents = discover_agents(packages=["nonexistent_package_xyz"])
        assert agents == []


# ============================================================
# TestResolvePrompt -- template variable substitution
# ============================================================


class TestResolvePrompt:
    def test_resolve_basic(self):
        result = resolve_prompt("Hello {name}", {"name": "World"})
        assert result == "Hello World"

    def test_resolve_all_provided(self):
        template = "Agent {agent} runs on {host} with {mode} mode"
        variables = {"agent": "explorer", "host": "localhost", "mode": "debug"}
        result = resolve_prompt(template, variables)
        assert result == "Agent explorer runs on localhost with debug mode"

    def test_resolve_missing_variable(self):
        with pytest.raises(ValueError, match="city"):
            resolve_prompt("Hello {name} from {city}", {"name": "World"})

    def test_resolve_curly_braces_in_value(self):
        """JSON in a substituted value is not treated as template variables."""
        result = resolve_prompt(
            "Config: {config}",
            {"config": '{"key": "value"}'},
        )
        assert result == 'Config: {"key": "value"}'

    def test_resolve_empty_variables(self):
        """Template with no placeholders and empty variables returns unchanged."""
        result = resolve_prompt("No placeholders here", {})
        assert result == "No placeholders here"

    def test_resolve_no_variables_but_has_placeholders(self):
        """Placeholders with empty variables dict raises ValueError."""
        with pytest.raises(ValueError, match="name"):
            resolve_prompt("Hello {name}", {})

    def test_resolve_repeated_placeholder(self):
        """Same placeholder appearing twice is replaced in both occurrences."""
        result = resolve_prompt("{x} and {x}", {"x": "val"})
        assert result == "val and val"
