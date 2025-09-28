import logging
import logging.config
import sys
from typing import Any

import structlog
from pythonjsonlogger import json as logging_json

from src.config import get_settings


def add_service_info(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Add service and environment information to the log event dictionary.

    Args:
        _ (Any): Unused.
        __ (Any): Unused.
        event_dict (dict[str, Any]): The log event dictionary.

    Returns:
        dict[str, Any]: The modified log event dictionary with service and environment info.
    """
    event_dict.setdefault("service", "smo-sim-api")
    event_dict.setdefault("env", get_settings().service.env)
    return event_dict


def add_log_level(
    _, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Add the log level to the log event dictionary.

    Args:
        _ (Any): Unused.
        method_name (str): The name of the logging method (e.g., "info", "debug").
        event_dict (dict[str, Any]): The log event dictionary.

    Returns:
        dict[str, Any]: The modified log event dictionary with the log level.
    """
    event_dict["level"] = method_name.upper()
    return event_dict


def add_timestamp(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Adds an ISO-formatted timestamp to the log event dictionary.

    Args:
        _ (Any): Unused.
        __ (Any): Unused.
        event_dict (dict[str, Any]): The log event dictionary.

    Returns:
        dict[str, Any]: The modified log event dictionary with the timestamp.
    """
    event_dict["timestamp"] = structlog.processors.TimeStamper(fmt="iso")(
        None, "", {}
    )["timestamp"]
    return event_dict


def add_logger_name(
    logger: logging.Logger, _, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Add the logger's name to the log event dictionary.

    Args:
        logger (logging.Logger): The logger instance.
        _ (Any): Unused.
        event_dict (dict[str, Any]): The log event dictionary.

    Returns:
        dict[str, Any]: The modified log event dictionary with the logger's name.
    """
    if logger:
        event_dict["logger"] = logger.name
    return event_dict


def add_callsite(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Add callsite information (module, function, line number) to the log event dictionary.

    Args:
        _ (Any): Unused.
        __ (Any): Unused.
        event_dict (dict[str, Any]): The log event dictionary.

    Returns:
        dict[str, Any]: The modified log event dictionary with callsite information.
    """
    record = event_dict.get("_record")
    if record:
        event_dict["module"] = record.module
        event_dict["func"] = record.funcName
        event_dict["line"] = record.lineno
    return event_dict


def configure_logging() -> None:
    """
    Configures the application's logging using structlog.

    This function sets up a shared set of processors for structlog,
    configures a console renderer (JSON for production, pretty for dev),
    and sets up a StreamHandler with a custom formatter for root logger.
    """
    log_level = get_settings().logger_settings.log_level.upper()
    is_dev = (
        get_settings().service.env == "dev"
        and not get_settings().logger_settings.log_json
    )

    if is_dev:
        console_renderer = structlog.dev.ConsoleRenderer(
            colors=True, sort_keys=False
        )
    else:
        console_renderer = structlog.processors.JSONRenderer(
            serializer=lambda obj: logging_json.dumps(obj, ensure_ascii=False)
        )

    logging.basicConfig(level=log_level, stream=sys.stdout)
    for noisy in [
        "uvicorn.error",
        "uvicorn.access",
        "gunicorn.error",
        "asyncio",
    ]:
        logging.getLogger(noisy).setLevel(
            logging.INFO if is_dev else logging.WARN
        )

    shared_processors = [
        add_service_info,
        add_log_level,
        add_timestamp,
        add_logger_name,
        add_callsite,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            *shared_processors,
            console_renderer,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )

    class StructlogJSONFormatter(logging.Formatter):
        def format(self, record):
            event_dict = {
                "event": record.getMessage(),
                "_record": record,
            }
            for proc in shared_processors:
                event_dict = proc(
                    logging.getLogger(record.name),
                    record.levelname.lower(),
                    event_dict,
                )
            if isinstance(console_renderer, structlog.processors.JSONRenderer):
                return console_renderer(event_dict)
            else:
                return console_renderer(event_dict)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level, logging.INFO))
    handler.setFormatter(StructlogJSONFormatter())
    root = logging.getLogger()

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)


def bind_request_context(
    request_id: str | None = None, user_id: str | None = None
) -> None:
    """
    Bind request-specific context variables for structured logging.

    Args:
        request_id (str | None): The ID of the current request. Defaults to None.
        user_id (str | None): The ID of the current user. Defaults to None.
    """
    structlog.contextvars.clear_contextvars()
    if request_id:
        structlog.contextvars.bind_contextvars(request_id=request_id)
    if user_id:
        structlog.contextvars.bind_contextvars(user_id=user_id)
