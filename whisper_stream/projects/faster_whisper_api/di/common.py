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
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Final, cast

from dependency_injector import containers, providers
from dotenv import load_dotenv

from whisper_stream.projects.faster_whisper_api.config.faster_whisper_api_launch_config import (
    FasterWhisperAPILaunchConfig,
)
from whisper_stream.projects.faster_whisper_api.config.faster_whisper_model_config import (
    FasterWhisperAPIModelFactoryConfig,
)
from whisper_stream.projects.faster_whisper_api.config.faster_whisper_transcription_config import (
    FasterWhisperAPITranscriptionConfig,
)
from whisper_stream.projects.faster_whisper_api.logger import (
    APIBoundLogger,
    get_api_logger,
)

from whisper_stream.core.logger import get_log_level_name_from_string


@dataclass(frozen=True, slots=True)
class AppMetaData:
    application: Final[str] = "whisper-stream-faster-whisper-api"
    full_name: Final[str] = "whisper-stream-faster-whisper-api:backend"
    description: Final[str] = "serving app for fast-whisper"
    version: Final[str] = version("whisper-stream-faster-whisper-api")


class CommonServiceContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    env = providers.Resource(load_dotenv, dotenv_path=Path.cwd() / ".env", verbose=True)

    wiring_config = containers.WiringConfiguration(
        modules=[
            "whisper_stream.projects.faster_whisper_api.preload.faster_whisper_model",
            "whisper_stream.projects.faster_whisper_api.di.faster_whisper_model",
            "whisper_stream.projects.faster_whisper_api.endpoints.probes",
            "whisper_stream.projects.faster_whisper_api.endpoints.transcribe",
            "whisper_stream.projects.faster_whisper_api.app",
            "whisper_stream.projects.faster_whisper_api.launcher",
            "whisper_stream.projects.faster_whisper_api.__main__",
        ],
        auto_wire=True,
    )

    app_metadata = providers.ThreadSafeSingleton(AppMetaData)
    model_config = providers.ThreadSafeSingleton(FasterWhisperAPIModelFactoryConfig)
    transcription_config = providers.ThreadSafeSingleton(
        FasterWhisperAPITranscriptionConfig
    )
    launch_config = providers.ThreadSafeSingleton(FasterWhisperAPILaunchConfig)

    logger: providers.ThreadSafeSingleton[
        APIBoundLogger
    ] = providers.ThreadSafeSingleton(
        get_api_logger,
        scope=cast(str, config().get("scope", app_metadata().full_name)),
        application_name=app_metadata().application,
        min_log_level=get_log_level_name_from_string(
            cast(str, config().get("log_level", "INFO"))
        ),
        binds=cast(dict[str, str], config().get("binds", {})),
    )


__all__: list[str] = ["CommonServiceContainer"]
