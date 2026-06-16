from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepStart:
    session_id: str
    message_id: str
    timestamp: int


@dataclass(frozen=True)
class Text:
    session_id: str
    message_id: str
    text: str
    timestamp: int


@dataclass(frozen=True)
class ToolUse:
    session_id: str
    message_id: str
    tool: str
    call_id: str
    status: str  # "completed", "error", etc.
    input: dict
    output: str
    title: str
    timestamp: int


@dataclass(frozen=True)
class StepFinish:
    session_id: str
    message_id: str
    reason: str  # "stop", "tool-calls"
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost: float
    timestamp: int


@dataclass(frozen=True)
class Error:
    session_id: str
    name: str
    message: str
    data: dict
    timestamp: int


@dataclass(frozen=True)
class Result:
    session_id: str
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    steps: int
