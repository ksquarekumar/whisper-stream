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
from typing import TypeAlias
from whisper_stream.core.logger import (
    BoundLogger,
    LogLevelNames,
    get_context_vars,
    get_application_logger,
)


APIBoundLogger: TypeAlias = BoundLogger


def get_api_logger(
    scope: str,
    application_name: str = "whisper-stream:faster-whiser-api",
    min_log_level: LogLevelNames = "INFO",
    binds: dict[str, str] | None = None,
) -> BoundLogger:
    _logger_binds: dict[str, str] = binds or {}
    # get_context_vars
    context_vars: dict[str, str] = get_context_vars(application_name=application_name)
    # Create a new instance of `BoundLogger` class
    instance: BoundLogger = get_application_logger(
        scope=scope,
        application_name=application_name,
        min_log_level=min_log_level,
        context_vars=context_vars,
        binds=_logger_binds,
    )
    return instance


__all__: list[str] = ["APIBoundLogger", "get_api_logger"]
