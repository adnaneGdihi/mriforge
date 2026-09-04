import io
import json
import logging

from spectramr.infrastructure.services.logging_service import JSONFormatter, LoggingService


def test_json_formatter():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_path.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.custom_field = "custom_value"

    json_output = formatter.format(record)
    data = json.loads(json_output)

    assert data["message"] == "Test message"
    assert data["level"] == "INFO"
    assert data["name"] == "test_logger"
    assert data["custom_field"] == "custom_value"
    assert "timestamp" in data


def test_logging_service_json_output():
    # Setup
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())

    service = LoggingService("test_service")
    service.logger.addHandler(handler)
    service.logger.setLevel(logging.INFO)

    # Action
    service.log_info("Info message", extra={"user_id": 123})

    # Verify
    output = stream.getvalue().strip()
    data = json.loads(output)

    assert data["message"] == "Info message"
    assert data["level"] == "INFO"
    assert data["user_id"] == 123
