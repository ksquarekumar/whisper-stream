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
from enum import IntEnum
from importlib.metadata import version
from platform import architecture, python_version
import sys
from typing import Final, Literal, TypeAlias, TypeGuard
import orjson

import structlog
from structlog.stdlib import BoundLogger as StdLibBoundLogger
from structlog.types import Processor

BoundLogger: TypeAlias = StdLibBoundLogger

LogLevelNames = Literal[
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


NAMES_TO_LEVELS: dict[LogLevelNames, LogLevels] = {
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


def is_valid_log_level_name(name: str) -> TypeGuard[LogLevelNames]:
    return name in NAMES_TO_LEVELS


def get_log_level_name_from_string(name: str) -> LogLevelNames:
    _name: str = name.upper()
    return _name if is_valid_log_level_name(_name) else "INFO"


def get_log_level_int_from_name(name: str) -> LogLevels:
    return NAMES_TO_LEVELS[get_log_level_name_from_string(name=name)]


def get_context_vars(application_name: str = "whisper_stream") -> dict[str, str]:
    return {
        "application": application_name,
        "version": str(version(application_name)),
        "python_version": python_version(),
        "platform_architecture": str(architecture()),
    }


DEFAULT_TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
DEFAULT_CONTEXT_VARS: Final[dict[str, str]] = get_context_vars()
_LOGGERS: dict[tuple[str, str, str], BoundLogger] = {}


def _get_timestamper(
    fmt: str = DEFAULT_TIMESTAMP_FORMAT,
) -> structlog.processors.TimeStamper:
    return structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")


def _get_processors(
    fmt: str = DEFAULT_TIMESTAMP_FORMAT,
) -> list[structlog.types.Processor]:
    processors: list[Processor] = []
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        _get_timestamper(fmt=fmt),
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
            structlog.processors.KeyValueRenderer(
                key_order=["event", "time_taken"],
                drop_missing=True,
                repr_native_str=True,
            ),
        ]

    return processors


def get_application_logger(
    scope: str,
    application_name: str = "whisper_stream",
    min_log_level: LogLevelNames | None = "INFO",
    datetime_fmt: str = DEFAULT_TIMESTAMP_FORMAT,
    context_vars: dict[str, str] = DEFAULT_CONTEXT_VARS,
    binds: dict[str, str] | None = None,
) -> BoundLogger:
    # quicker fast path
    key: tuple[str, str, str] = (
        scope,
        application_name,
        orjson.dumps(context_vars).decode("utf-8"),
    )
    if key in _LOGGERS:
        return _LOGGERS[key]
    # do the work the we need to
    _binds: dict[str, str] = binds or {}
    _setup_logging(
        min_log_level=min_log_level,
        datetime_fmt=datetime_fmt,
        context_vars=context_vars,
    )

    _LOGGERS[key] = structlog.get_logger(
        name=scope,
        **_binds,
    )
    return _LOGGERS[key]


def _setup_logging(
    min_log_level: LogLevelNames | None = "INFO",
    datetime_fmt: str = DEFAULT_TIMESTAMP_FORMAT,
    context_vars: dict[str, str] = DEFAULT_CONTEXT_VARS,
) -> None:
    _min_log_level: LogLevels = get_log_level_int_from_name(
        min_log_level
        if min_log_level is not None
        else os.environ.get("LOG_LEVEL", "INFO")
    )
    logging.basicConfig(level=_min_log_level.value)
    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            min_level=_min_log_level.value
        ),
        processors=_get_processors(fmt=datetime_fmt),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(**context_vars)


all: list[str] = [
    "BoundLogger",
    "LogLevelNames",
    "LogLevels",
    "NAMES_TO_LEVELS",
    "is_valid_log_level_name",
    "get_log_level_name_from_string",
    "get_log_level_int_from_name" "get_context_vars",
    "get_application_logger",
    "DEFAULT_TIMESTAMP_FORMAT",
]
