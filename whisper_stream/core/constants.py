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


def create_package_directories_if_not_exists(
    scope: str, path_or_paths: Path | list[Path]
) -> None:
    _paths: list[Path] = (
        path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
    )
    _paths_to_create = [path for path in _paths if not path.exists()]

    if len(_paths_to_create) > 0:
        _logger: Final[BoundLogger] = get_application_logger(scope)
        for _path in _paths_to_create:
            _logger.info(f"creating directories for package", path=_path)
            _path.mkdir(exist_ok=True)
        _logger.info("finished setting up package directories")


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
create_package_directories_if_not_exists(
    scope="whisper-steam:core:init",
    path_or_paths=[PACKAGE_COMMON_PATH_PREFIX, CACHE_PATH_PREFIX, DEFAULT_DATA_PATH],
)
# side-effects done

__all__: list[str] = [
    "create_package_directories_if_not_exists",
    "PACKAGE_COMMON_PATH_PREFIX",
    "CACHE_PATH_PREFIX",
    "DEFAULT_DATA_PATH",
    "WhisperValidCheckpoints",
    "WhisperValidTasks",
]
