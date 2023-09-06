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
from typing import Final

from dependency_injector import containers, providers
from dotenv import load_dotenv

from faster_whisper_api.config.faster_whisper_api_launch_config import (
    FasterWhisperAPILaunchConfig,
)
from faster_whisper_api.config.faster_whisper_model_config import (
    FasterWhisperAPIModelFactoryConfig,
)
from faster_whisper_api.config.faster_whisper_transcription_config import (
    FasterWhisperAPITranscriptionConfig,
)
from faster_whisper_api.logger import APILogger


@dataclass(frozen=True, slots=True)
class AppMetaData:
    application: Final[str] = "faster-whisper-api"
    full_name: Final[str] = "faster-whisper-api:backend"
    version: Final[str] = version("faster-whisper-api")
    description: Final[str] = "serving app for fast-whisper"


class CommonServiceContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    env = providers.Resource(load_dotenv)

    wiring_config = containers.WiringConfiguration(
        modules=[
            "faster_whisper_api.preload.faster_whisper_model",
            "faster_whisper_api.di.faster_whisper_model",
            "faster_whisper_api.endpoints.probes",
            "faster_whisper_api.endpoints.transcribe",
            "faster_whisper_api.launcher",
            "faster_whisper_api.app",
        ],
        auto_wire=True,
    )

    app_metadata = providers.ThreadSafeSingleton(AppMetaData)
    model_config = providers.ThreadSafeSingleton(FasterWhisperAPIModelFactoryConfig)
    transcription_config = providers.ThreadSafeSingleton(
        FasterWhisperAPITranscriptionConfig
    )
    launch_config = providers.ThreadSafeSingleton(FasterWhisperAPILaunchConfig)

    logger: providers.ThreadSafeSingleton[APILogger] = providers.ThreadSafeSingleton(
        APILogger,
        application=app_metadata().application,
        version=app_metadata().version,
        scope=app_metadata().full_name,
        context_vars=config.context,
    )


__all__: list[str] = ["CommonServiceContainer"]
