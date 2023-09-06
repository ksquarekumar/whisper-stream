#
# # Copyright © 2023 krishnakumar <ksquarekumar@gmail.com>.
# #
# # Licensed under the Apache License, Version 2.0 (the "License"). You
# # may not use this file except in compliance with the License. A copy of
# # the License is located at:
# #
# # https://github.com/ksquarekumar/whisper-stream/blob/main/LICENSE
# #
# # or in the "license" file accompanying this file. This file is
# # distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# # ANY KIND, either express or implied. See the License for the specific
# # language governing permissions and limitations under the License.
# #
# # This file is part of the whisper-stream.
# # see (https://github.com/ksquarekumar/whisper-stream)
# #
# # SPDX-License-Identifier: Apache-2.0
# #
# # You should have received a copy of the APACHE LICENSE, VERSION 2.0
# # along with this program. If not, see <https://apache.org/licenses/LICENSE-2.0>
#
import logging
import os
import sys
from enum import IntEnum
from importlib.metadata import version
from platform import architecture, python_version
from typing import Any, Literal, NewType, TypeGuard

import orjson
import structlog
from structlog.types import Processor

BoundLogger = NewType("BoundLogger", structlog._config.BoundLoggerLazyProxy)

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


NAMES_TO_LEVELS: dict[LOG_LEVEL_NAMES, LogLevels] = {
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
    return name.upper() in NAMES_TO_LEVELS


def get_log_level_name(name: str) -> LOG_LEVEL_NAMES:
    _name: str = name.upper()
    return _name if _is_level(_name) else "INFO"


def get_log_level_from_name(name: str) -> LogLevels:
    return NAMES_TO_LEVELS[get_log_level_name(name=name)]


def _get_processors() -> list[structlog.types.Processor]:
    processors: list[Processor] = []
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if sys.stdout.isatty():
        # Pretty printing when we run in a terminal session.
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(repr_native_str=True),
        ]
    else:
        # Print JSON when we run in production
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(
                serializer=orjson.dumps,
                key_order=["event", "time_taken"],
                drop_missing=True,
                repr_native_str=True,
            ),
        ]

    return processors


CONTEXT_VARS = {
    "application": "whisper_stream",
    "version": str(version("whisper_stream")),
    "python_version": python_version(),
    "platform_architecture": architecture(),
}

_LOGGERS: dict[str, BoundLogger] = {}


def get_application_logger(
    name: str,
    min_log_level: LOG_LEVEL_NAMES | None = "INFO",
    binds: dict[str, Any] | None = None,
    context_vars: dict[str, Any] | None = None,
) -> BoundLogger:
    # quicker fast path
    if name in _LOGGERS:
        return _LOGGERS[name]
    if not structlog.is_configured():
        setup_logging(min_log_level=min_log_level)
    _binds = binds or {}
    _context_vars = context_vars or CONTEXT_VARS
    _LOGGERS[name] = BoundLogger(
        structlog.get_logger(
            name,
            {
                **_context_vars,
                **_binds,
            },
        )
    )
    return _LOGGERS[name]


def setup_logging(min_log_level: LOG_LEVEL_NAMES | None = "INFO") -> None:
    _log_level: LogLevels = get_log_level_from_name(
        name=min_log_level if min_log_level else os.environ.get("LOG_LEVEL", "INFO")
    )
    logging.basicConfig(level=_log_level.value)
    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(min_level=_log_level),
        processors=_get_processors(),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(**CONTEXT_VARS)


__all__: list[str] = [
    "BoundLogger",
    "get_application_logger",
    "LOG_LEVEL_NAMES",
    "NAMES_TO_LEVELS",
]
