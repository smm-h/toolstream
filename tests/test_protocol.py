import json

from openstream._protocol import parse_event
from openstream.events import Error, StepFinish, StepStart, Text, ToolUse


class TestParseStepStart:
    def test_basic(self):
        line = json.dumps({
            "type": "step_start",
            "timestamp": 1000,
            "sessionID": "ses_abc123",
            "part": {
                "type": "step-start",
                "messageID": "msg_def456",
                "snapshot": "...",
            },
        })
        event = parse_event(line)
        assert isinstance(event, StepStart)
        assert event.session_id == "ses_abc123"
        assert event.message_id == "msg_def456"
        assert event.timestamp == 1000


class TestParseText:
    def test_basic(self):
        line = json.dumps({
            "type": "text",
            "timestamp": 2000,
            "sessionID": "ses_abc123",
            "part": {
                "type": "text",
                "messageID": "msg_def456",
                "text": "Hello, world!",
                "time": {"start": 1, "end": 2},
                "metadata": {},
            },
        })
        event = parse_event(line)
        assert isinstance(event, Text)
        assert event.text == "Hello, world!"
        assert event.session_id == "ses_abc123"
        assert event.timestamp == 2000


class TestParseToolUse:
    def test_basic(self):
        line = json.dumps({
            "type": "tool_use",
            "timestamp": 3000,
            "sessionID": "ses_abc123",
            "part": {
                "type": "tool",
                "tool": "read",
                "messageID": "msg_def456",
                "callID": "call_xyz789",
                "state": {
                    "status": "completed",
                    "input": {"path": "/etc/hostname"},
                    "output": "myhost\n",
                    "metadata": {},
                },
                "title": "/etc/hostname",
            },
        })
        event = parse_event(line)
        assert isinstance(event, ToolUse)
        assert event.tool == "read"
        assert event.call_id == "call_xyz789"
        assert event.status == "completed"
        assert event.input == {"path": "/etc/hostname"}
        assert event.output == "myhost\n"
        assert event.title == "/etc/hostname"


class TestParseStepFinish:
    def test_basic(self):
        line = json.dumps({
            "type": "step_finish",
            "timestamp": 4000,
            "sessionID": "ses_abc123",
            "part": {
                "type": "step-finish",
                "messageID": "msg_def456",
                "reason": "stop",
                "tokens": {
                    "total": 150,
                    "input": 100,
                    "output": 40,
                    "reasoning": 10,
                    "cache": {"write": 5, "read": 20},
                },
                "cost": 0.0015,
            },
        })
        event = parse_event(line)
        assert isinstance(event, StepFinish)
        assert event.reason == "stop"
        assert event.input_tokens == 100
        assert event.output_tokens == 40
        assert event.reasoning_tokens == 10
        assert event.cache_read_tokens == 20
        assert event.cache_write_tokens == 5
        assert event.cost == 0.0015

    def test_tool_calls_reason(self):
        line = json.dumps({
            "type": "step_finish",
            "timestamp": 4000,
            "sessionID": "ses_abc123",
            "part": {
                "type": "step-finish",
                "messageID": "msg_def456",
                "reason": "tool-calls",
                "tokens": {
                    "total": 200,
                    "input": 150,
                    "output": 50,
                    "reasoning": 0,
                    "cache": {"write": 0, "read": 0},
                },
                "cost": 0.002,
            },
        })
        event = parse_event(line)
        assert isinstance(event, StepFinish)
        assert event.reason == "tool-calls"


class TestParseError:
    def test_basic(self):
        line = json.dumps({
            "type": "error",
            "timestamp": 5000,
            "sessionID": "ses_abc123",
            "error": {
                "name": "rate_limit_error",
                "message": "Rate limit exceeded",
                "data": {"retry_after": 30},
            },
        })
        event = parse_event(line)
        assert isinstance(event, Error)
        assert event.name == "rate_limit_error"
        assert event.message == "Rate limit exceeded"
        assert event.data == {"retry_after": 30}


class TestMalformedInput:
    def test_empty_line(self):
        assert parse_event("") is None

    def test_whitespace(self):
        assert parse_event("   \n") is None

    def test_invalid_json(self):
        assert parse_event("not json at all") is None

    def test_json_array(self):
        assert parse_event("[1, 2, 3]") is None

    def test_unknown_type(self):
        assert parse_event('{"type": "unknown_event", "data": {}}') is None

    def test_missing_type(self):
        assert parse_event('{"sessionID": "ses_abc", "timestamp": 100}') is None

    def test_status_line(self):
        # opencode sometimes prints non-JSON status lines
        assert parse_event("Starting opencode...") is None
