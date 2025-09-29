import logging
import structlog
from unittest.mock import MagicMock

from src.core.logging import (
    add_log_level,
    add_timestamp,
    add_logger_name,
    add_callsite,
    bind_request_context,
)


def test_add_log_level():
    """Test that the log level is correctly added to the event dictionary."""
    event_dict = add_log_level(None, "info", {})
    assert event_dict == {"level": "INFO"}


def test_add_timestamp():
    """Test that a timestamp is added to the event dictionary."""
    event_dict = add_timestamp(None, None, {})
    assert "timestamp" in event_dict
    assert isinstance(event_dict["timestamp"], str)


def test_add_logger_name():
    """Test that the logger's name is added to the event dictionary."""
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.name = "test_logger"
    event_dict = add_logger_name(mock_logger, None, {})
    assert event_dict == {"logger": "test_logger"}


def test_add_callsite():
    """Test that callsite info from the log record is added."""
    mock_record = MagicMock()
    mock_record.module = "test_module"
    mock_record.funcName = "test_func"
    mock_record.lineno = 42
    event_dict = {"_record": mock_record}

    result_dict = add_callsite(None, None, event_dict)

    assert result_dict["module"] == "test_module"
    assert result_dict["func"] == "test_func"
    assert result_dict["line"] == 42


def test_bind_request_context():
    """Test binding and clearing of context variables."""
    structlog.contextvars.clear_contextvars()

    try:
        bind_request_context(request_id="123", user_id="abc")
        context = structlog.contextvars.get_contextvars()
        assert context == {"request_id": "123", "user_id": "abc"}

        bind_request_context(request_id="456")
        context = structlog.contextvars.get_contextvars()
        assert context == {"request_id": "456"}
        assert "user_id" not in context

    finally:
        structlog.contextvars.clear_contextvars()
