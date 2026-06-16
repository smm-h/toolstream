"""Shared fixtures for llmloop tests -- mock LLM responder."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llmloop.config import SessionConfig


def text_response(
    content: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict:
    """Build a canned text-only chat completion response."""
    return {
        "choices": [{
            "message": {
                "content": content,
                "tool_calls": None,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def tool_call_response(
    tool_name: str,
    arguments: dict,
    call_id: str = "call_1",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict:
    """Build a canned tool-call chat completion response."""
    return {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def direct_config(**overrides: Any) -> SessionConfig:
    """Build a SessionConfig suitable for DirectClient with mock http_client."""
    defaults: dict[str, Any] = dict(
        model="test-model",
        backend="direct",
        api_key="test-key",
        base_url="https://mock-gateway.test/v1/chat/completions",
    )
    defaults.update(overrides)
    return SessionConfig(**defaults)


@pytest.fixture
def mock_llm_responses():
    """Factory fixture: call with canned response dicts to get a mock httpx.AsyncClient."""

    def factory(*responses: dict) -> httpx.AsyncClient:
        it = iter(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            try:
                response_dict = next(it)
            except StopIteration:
                raise RuntimeError(
                    f"mock_llm_responses exhausted: expected at most "
                    f"{len(responses)} call(s)"
                )
            return httpx.Response(200, json=response_dict)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory
