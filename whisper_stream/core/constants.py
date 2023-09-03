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
from pathlib import Path
from typing import Final, Literal

from whisper_stream.core.logger import get_application_logger, BoundLogger


def create_package_directories(
    path_or_paths: Path | list[Path], logger: BoundLogger
) -> None:
    _paths: list[Path] = (
        path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
    )
    for _path in _paths:
        logger.info(f"creating directories for package", path=_path)
        _path.mkdir(exist_ok=True)


PACKAGE_COMMON_PATH_PREFIX: Final[Path] = Path.home() / ".whisper_stream"
CACHE_PATH_PREFIX: Final[Path] = Path(PACKAGE_COMMON_PATH_PREFIX / ".cache")
DEFAULT_DATA_PATH: Final[Path] = Path(PACKAGE_COMMON_PATH_PREFIX / ".data")

WhisperValidCheckpoints = Literal[
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
    "openai/whisper-large-v2",
]

WhisperValidTasks = Literal["transcribe", "translate"]

# import side-effects
_scope: Final[str] = "whisper-steam:core:init"
_temp_logger: Final[BoundLogger] = get_application_logger(_scope)


create_package_directories(
    [PACKAGE_COMMON_PATH_PREFIX, CACHE_PATH_PREFIX, DEFAULT_DATA_PATH], _temp_logger
)
_temp_logger.info("finished setting up package directories")

del _temp_logger, _scope
# side-effects done

__all__: list[str] = [
    "create_package_directories",
    "PACKAGE_COMMON_PATH_PREFIX",
    "CACHE_PATH_PREFIX",
    "DEFAULT_DATA_PATH",
    "WhisperValidCheckpoints",
    "WhisperValidTasks",
]
