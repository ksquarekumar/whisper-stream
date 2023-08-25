from enum import IntEnum
import os
from platform import python_version, architecture
import sys
from typing import Any, Literal, NewType, TypeGuard
import orjson
import structlog
from structlog.types import Processor
import importlib_metadata
import logging

BoundLogger = NewType("BoundLogger", structlog.stdlib.BoundLogger)

LOG_LEVEL_NAMES = Literal[
    "NOTSET",
    "DEBUG",
    "INFO",
    "WARNING",
    "WARN",
    "ERROR",
    "EXCEPTION",
    "FATAL",
    "CRITICAL",
]


class LogLevels(IntEnum):
    CRITICAL = 50
    FATAL = 50
    EXCEPTION = 40
    ERROR = 40
    WARNING = 30
    WARN = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0


_NAME_TO_LEVEL: dict[LOG_LEVEL_NAMES, LogLevels] = {
    "NOTSET": LogLevels.NOTSET,
    "DEBUG": LogLevels.DEBUG,
    "INFO": LogLevels.INFO,
    "WARNING": LogLevels.WARNING,
    "WARN": LogLevels.WARN,
    "ERROR": LogLevels.ERROR,
    "EXCEPTION": LogLevels.EXCEPTION,
    "FATAL": LogLevels.FATAL,
    "CRITICAL": LogLevels.CRITICAL,
}


def _is_level(name: str) -> TypeGuard[LOG_LEVEL_NAMES]:
    return name.upper() in _NAME_TO_LEVEL


def get_log_level_name(name: str) -> LOG_LEVEL_NAMES:
    _name: str = name.upper()
    return _name if _is_level(_name) else "INFO"


def get_log_level_from_name(name: str) -> LogLevels:
    return _NAME_TO_LEVEL[get_log_level_name(name=name)]


def _get_processors() -> list[structlog.types.Processor]:
    processors: list[Processor] = []
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if sys.stdout.isatty():
        # Pretty printing when we run in a terminal session.
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        # Print JSON when we run in production
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(serializer=orjson.dumps),
        ]

    return processors


CONTEXT_VARS = {
    "application": "whisper_stream",
    "version": str(importlib_metadata.version("whisper_stream")),  # type: ignore[no-untyped-call]
    "python_version": python_version(),
    "platform_architecture": architecture(),
}

_LOGGERS: dict[str, BoundLogger] = {}


def get_application_logger(name: str, binds: dict[str, Any] | None = None) -> BoundLogger:
    # quicker fast path
    if name in _LOGGERS:
        return _LOGGERS[name]
    if not structlog.is_configured():
        setup_logging()
    _binds = binds or {}
    _LOGGERS[name] = BoundLogger(structlog.get_logger(**_binds, **CONTEXT_VARS))
    return _LOGGERS[name]


def setup_logging(setup_snowflake: Literal["create", "skip"] = "create") -> None:
    _log_level: LogLevels = get_log_level_from_name(name=os.environ.get("LOG_LEVEL", "INFO"))
    logging.basicConfig(level=_log_level.value)
    structlog.configure(
        processors=_get_processors(),
        wrapper_class=structlog.make_filtering_bound_logger(min_level=_log_level),
    )
    structlog.contextvars.bind_contextvars(**CONTEXT_VARS)


__all__: list[str] = ["BoundLogger", "get_application_logger"]
