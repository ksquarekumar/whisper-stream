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
from whisper_stream.core.logger import (
    BoundLogger,
    LOG_LEVEL_NAMES,
    get_application_logger,
)


class APILogger:
    def __new__(
        cls,
        application: str,
        version: str,
        scope: str,
        min_log_level: LOG_LEVEL_NAMES = "INFO",
        context_vars: dict[str, str] | None = None,
    ) -> BoundLogger:
        _context_vars: dict[str, str] = context_vars or {}
        _logger_binds: dict[str, str] = {
            "application": application,
            "version": version,
            **_context_vars,
        }
        # Create a new instance of `BoundLogger` class
        instance: BoundLogger = BoundLogger(
            get_application_logger(
                scope, min_log_level=min_log_level, binds=_logger_binds
            )
        )
        return instance


__all__: list[str] = ["APILogger"]
