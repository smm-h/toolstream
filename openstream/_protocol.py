from __future__ import annotations

import json

from .events import Error, StepFinish, StepStart, Text, ToolUse

Event = StepStart | Text | ToolUse | StepFinish | Error


def parse_event(line: str) -> Event | None:
    """Parse a single NDJSON line into a typed event.

    Returns None for unparseable or unrecognized lines.
    """
    line = line.strip()
    if not line:
        return None

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    event_type = data.get("type")
    session_id = data.get("sessionID", "")
    timestamp = data.get("timestamp", 0)
    part = data.get("part", {})

    if event_type == "step_start":
        return StepStart(
            session_id=session_id,
            message_id=part.get("messageID", ""),
            timestamp=timestamp,
        )

    if event_type == "text":
        return Text(
            session_id=session_id,
            message_id=part.get("messageID", ""),
            text=part.get("text", ""),
            timestamp=timestamp,
        )

    if event_type == "tool_use":
        state = part.get("state", {})
        return ToolUse(
            session_id=session_id,
            message_id=part.get("messageID", ""),
            tool=part.get("tool", ""),
            call_id=part.get("callID", ""),
            status=state.get("status", ""),
            input=state.get("input", {}),
            output=state.get("output", ""),
            title=part.get("title", ""),
            timestamp=timestamp,
        )

    if event_type == "step_finish":
        tokens = part.get("tokens", {})
        cache = tokens.get("cache", {})
        return StepFinish(
            session_id=session_id,
            message_id=part.get("messageID", ""),
            reason=part.get("reason", ""),
            input_tokens=tokens.get("input", 0),
            output_tokens=tokens.get("output", 0),
            reasoning_tokens=tokens.get("reasoning", 0),
            cache_read_tokens=cache.get("read", 0),
            cache_write_tokens=cache.get("write", 0),
            cost=part.get("cost", 0.0),
            timestamp=timestamp,
        )

    if event_type == "error":
        error = data.get("error", {})
        return Error(
            session_id=session_id,
            name=error.get("name", ""),
            message=error.get("message", ""),
            data=error.get("data", {}),
            timestamp=timestamp,
        )

    return None
