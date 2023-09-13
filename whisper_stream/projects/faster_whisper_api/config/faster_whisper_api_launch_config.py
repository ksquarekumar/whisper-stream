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
import os
from ipaddress import AddressValueError
from pathlib import Path
from typing import Any

from pydantic import Field, IPvAnyInterface, ValidationError, validator
from uvicorn.config import (
    HTTPProtocolType,
    InterfaceType,
    LifespanType,
    LoopSetupType,
    WSProtocolType,
)
from whisper_stream.core.logger import NAMES_TO_LEVELS

from whisper_stream.projects.faster_whisper_api.config.base import APIBaseSettings


def log_level_default_factory(value: str | None = None) -> str:
    _value: str = value if value is not None else os.environ.get("LOG_LEVEL", "INFO")
    if _value.upper() in NAMES_TO_LEVELS:
        return _value.lower()
    invalid_log_level_error: str = (
        f"{_value} is not a valid log level, must be one of {NAMES_TO_LEVELS.keys()}"
    )
    raise ValueError(invalid_log_level_error)


def reload_opt_factory() -> bool:
    match os.environ.get("APP_ENV", os.environ.get("app_env", None)):
        case None | "prod" | "production":
            return False
        case _:
            return True


class FasterWhisperAPILaunchConfig(APIBaseSettings):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, gt=0, le=65535)
    env_file: str | None = Field(
        default=str(Path.cwd() / ".env"), validation_alias="dotenv_path"
    )
    log_config: str | None = Field(default=str(Path.cwd() / "log_config.yaml"))
    workers: int = Field(default=1, ge=-1, validation_alias="num_workers")
    loop: LoopSetupType = "uvloop"
    timeout_graceful_shutdown: int = Field(
        default=30, gt=15, validation_alias="worker_timeout"
    )
    limit_max_requests: int = Field(
        default=1000, ge=1, validation_alias="max_worker_requests"
    )
    limit_concurrency: int | None = Field(
        default=None, validation_alias="default_concurrency"
    )
    timeout_keep_alive: int = Field(
        default=15, ge=5, validation_alias="keep_alive_timeout"
    )
    access_log: bool = Field(default=True, validation_alias="enable_access_log")
    interface: InterfaceType = Field(default="asgi3")
    ws: WSProtocolType = Field(default="none", validation_alias="enable_websockets")
    ws_max_size: int = Field(default=16777216, ge=2048, multiple_of=2048)
    ws_max_queue: int = Field(default=32, ge=1)
    ws_ping_interval: float = Field(default=20.0, ge=1.0, multiple_of=0.1)
    ws_ping_timeout: float = Field(default=20.0, ge=1.0, multiple_of=0.1)
    ws_per_message_deflate: bool = Field(default=True)
    lifespan: LifespanType = Field(default="auto", validation_alias="enable_lifecycles")
    reload: bool = Field(default_factory=reload_opt_factory)
    http: HTTPProtocolType = Field(default="h11", validation_alias="httpspec")
    root_path: str = Field(default="")

    @validator("host")
    def validate_host_address(cls, v: Any) -> str:
        try:
            IPvAnyInterface(v)
            return str(v)
        except AddressValueError as exception:
            raise ValidationError(exception) from exception

    @validator("env_file", "log_config")
    def file_path_validator(cls, v: Any | None) -> str | None:
        if v is not None and Path(v).exists():
            return str(Path(v).absolute())
        else:
            return None

    @validator("workers")
    def validate_workers(cls, v: Any) -> int:
        if isinstance(v, int) and v > -1 and v != 0:
            return v
        error_message: str = f"{v} is not a valid workers value"
        raise ValidationError(error_message)

    @validator("limit_concurrency")
    def validate_concurrency(cls, v: Any) -> int | None:
        if v is None:
            return v
        if isinstance(v, int) and v > 1 and v != 0:
            return v
        error_message: str = f"{v} is not a valid concurrency value"
        raise ValidationError(error_message)


__all__: list[str] = ["FasterWhisperAPILaunchConfig"]
