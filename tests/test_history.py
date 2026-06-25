"""Tests for pluggable history strategies."""

from __future__ import annotations

from pathlib import Path

import pytest

from toolstream._direct import DirectClient
from toolstream._history import HistoryStrategy, TokenBudgetHistory, UnboundedHistory
from toolstream.config import SessionConfig


# --- UnboundedHistory ---


def test_unbounded_preserves_all():
    """100 messages appended are all returned by messages_for_api()."""
    h = UnboundedHistory()
    for i in range(100):
        h.append({"role": "user", "content": f"msg {i}"})
    msgs = h.messages_for_api()
    assert len(msgs) == 100
    assert msgs[0]["content"] == "msg 0"
    assert msgs[99]["content"] == "msg 99"


def test_unbounded_on_usage_is_noop():
    """on_usage does nothing on UnboundedHistory."""
    h = UnboundedHistory()
    h.append({"role": "system", "content": "sys"})
    h.on_usage({"total_tokens": 999999})
    assert len(h.messages_for_api()) == 1


# --- TokenBudgetHistory ---


def test_token_budget_drops_oldest():
    """When over 90% budget, oldest non-system messages are dropped."""
    h = TokenBudgetHistory(max_tokens=1000)
    h.append({"role": "system", "content": "system prompt"})
    for i in range(10):
        h.append({"role": "user", "content": f"msg {i}"})

    # Simulate usage at 95% of budget -> triggers trimming
    h.on_usage({"total_tokens": 950})
    msgs = h.messages_for_api()

    # System prompt must be first
    assert msgs[0]["role"] == "system"
    # Some non-system messages were dropped
    assert len(msgs) < 11


def test_token_budget_never_drops_system():
    """System message (index 0) is never dropped, even when far over budget."""
    h = TokenBudgetHistory(max_tokens=100)
    h.append({"role": "system", "content": "system prompt"})
    h.append({"role": "user", "content": "hello"})

    # Way over budget
    h.on_usage({"total_tokens": 200})
    msgs = h.messages_for_api()

    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "system prompt"


def test_token_budget_below_threshold():
    """Under 90% of budget, no messages are dropped."""
    h = TokenBudgetHistory(max_tokens=1000)
    h.append({"role": "system", "content": "system prompt"})
    for i in range(10):
        h.append({"role": "user", "content": f"msg {i}"})

    # 80% of budget -> under the 90% threshold
    h.on_usage({"total_tokens": 800})
    msgs = h.messages_for_api()

    # All 11 messages preserved (1 system + 10 user)
    assert len(msgs) == 11


def test_token_budget_no_usage_no_drop():
    """Before any on_usage call (total_tokens=0), no trimming occurs."""
    h = TokenBudgetHistory(max_tokens=1000)
    h.append({"role": "system", "content": "system prompt"})
    for i in range(5):
        h.append({"role": "user", "content": f"msg {i}"})
    msgs = h.messages_for_api()
    assert len(msgs) == 6


# --- Protocol conformance ---


def test_unbounded_is_history_strategy():
    """UnboundedHistory satisfies the HistoryStrategy protocol."""
    assert isinstance(UnboundedHistory(), HistoryStrategy)


def test_token_budget_is_history_strategy():
    """TokenBudgetHistory satisfies the HistoryStrategy protocol."""
    assert isinstance(TokenBudgetHistory(max_tokens=1000), HistoryStrategy)


# --- DirectClient integration ---


def test_direct_client_default_unbounded(tmp_path: Path):
    """DirectClient with no history arg uses UnboundedHistory."""
    config = SessionConfig(
        model="gpt-5.4",
        api_key="test-key",
        base_url="https://test-gateway.example.com",
        system_prompt="You are a test assistant.",
        cwd=str(tmp_path),
        auth_style="x-api-key",
    )
    client = DirectClient(config)
    assert isinstance(client._history, UnboundedHistory)


def test_direct_client_custom_history(tmp_path: Path):
    """DirectClient respects an explicitly passed history strategy."""
    config = SessionConfig(
        model="gpt-5.4",
        api_key="test-key",
        base_url="https://test-gateway.example.com",
        system_prompt="You are a test assistant.",
        cwd=str(tmp_path),
        auth_style="x-api-key",
    )
    history = TokenBudgetHistory(max_tokens=5000)
    client = DirectClient(config, history=history)
    assert client._history is history


def test_direct_client_config_history_strategy(tmp_path: Path):
    """DirectClient uses config.history_strategy when history arg is None."""
    history = TokenBudgetHistory(max_tokens=5000)
    config = SessionConfig(
        model="gpt-5.4",
        api_key="test-key",
        base_url="https://test-gateway.example.com",
        system_prompt="You are a test assistant.",
        cwd=str(tmp_path),
        auth_style="x-api-key",
        history_strategy=history,
    )
    client = DirectClient(config)
    assert client._history is history


def test_direct_client_history_arg_overrides_config(tmp_path: Path):
    """Explicit history arg takes precedence over config.history_strategy."""
    config_history = TokenBudgetHistory(max_tokens=5000)
    arg_history = UnboundedHistory()
    config = SessionConfig(
        model="gpt-5.4",
        api_key="test-key",
        base_url="https://test-gateway.example.com",
        system_prompt="You are a test assistant.",
        cwd=str(tmp_path),
        auth_style="x-api-key",
        history_strategy=config_history,
    )
    client = DirectClient(config, history=arg_history)
    assert client._history is arg_history
