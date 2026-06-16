from toolstream.events import Error, Result, StepFinish, StepStart, Text, ToolUse


class TestStepStart:
    def test_fields(self):
        e = StepStart(session_id="ses_1", message_id="msg_1", timestamp=100)
        assert e.session_id == "ses_1"
        assert e.message_id == "msg_1"
        assert e.timestamp == 100

    def test_frozen(self):
        e = StepStart(session_id="ses_1", message_id="msg_1", timestamp=100)
        try:
            e.session_id = "ses_2"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestText:
    def test_fields(self):
        e = Text(session_id="ses_1", message_id="msg_1", text="hello", timestamp=200)
        assert e.session_id == "ses_1"
        assert e.message_id == "msg_1"
        assert e.text == "hello"
        assert e.timestamp == 200


class TestToolUse:
    def test_fields(self):
        e = ToolUse(
            session_id="ses_1",
            message_id="msg_1",
            tool="read",
            call_id="call_1",
            status="completed",
            input={"path": "/etc/hostname"},
            output="myhost",
            title="/etc/hostname",
            timestamp=300,
        )
        assert e.tool == "read"
        assert e.call_id == "call_1"
        assert e.status == "completed"
        assert e.input == {"path": "/etc/hostname"}
        assert e.output == "myhost"
        assert e.title == "/etc/hostname"


class TestStepFinish:
    def test_fields(self):
        e = StepFinish(
            session_id="ses_1",
            message_id="msg_1",
            reason="stop",
            input_tokens=100,
            output_tokens=50,
            reasoning_tokens=10,
            cache_read_tokens=20,
            cache_write_tokens=5,
            cost=0.001,
            timestamp=400,
        )
        assert e.reason == "stop"
        assert e.input_tokens == 100
        assert e.output_tokens == 50
        assert e.reasoning_tokens == 10
        assert e.cache_read_tokens == 20
        assert e.cache_write_tokens == 5
        assert e.cost == 0.001


class TestError:
    def test_fields(self):
        e = Error(
            session_id="ses_1",
            name="rate_limit",
            message="Too many requests",
            data={"retry_after": 60},
            timestamp=500,
        )
        assert e.name == "rate_limit"
        assert e.message == "Too many requests"
        assert e.data == {"retry_after": 60}


class TestResult:
    def test_fields(self):
        r = Result(
            session_id="ses_1",
            total_input_tokens=200,
            total_output_tokens=100,
            total_cost=0.005,
            steps=2,
        )
        assert r.total_input_tokens == 200
        assert r.total_output_tokens == 100
        assert r.total_cost == 0.005
        assert r.steps == 2
