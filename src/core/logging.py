import logging
import logging.config
import sys
from typing import Any, Dict

import structlog
from pythonjsonlogger import json as logging_json

from src.config import get_settings


def add_service_info(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    event_dict.setdefault("service", "smo-sim-api")
    event_dict.setdefault("env", get_settings().service.env)
    return event_dict


def add_log_level(logger, method_name, event_dict):
    event_dict["level"] = method_name.upper()
    return event_dict


def add_timestamp(_, __, event_dict):
    event_dict["timestamp"] = structlog.processors.TimeStamper(fmt="iso")(
        None, "", {}
    )["timestamp"]
    return event_dict


def add_logger_name(logger, _, event_dict):
    if logger:
        event_dict["logger"] = logger.name
    return event_dict


def add_callsite(_, __, event_dict: dict):
    record = event_dict.get("_record")
    if record:
        event_dict["module"] = record.module
        event_dict["func"] = record.funcName
        event_dict["line"] = record.lineno
    return event_dict


def configure_logging() -> None:
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
    structlog.contextvars.clear_contextvars()
    if request_id:
        structlog.contextvars.bind_contextvars(request_id=request_id)
    if user_id:
        structlog.contextvars.bind_contextvars(user_id=user_id)
