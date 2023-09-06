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

from faster_whisper_api.config.faster_whisper_model_config import (
    FasterWhisperAPIModelFactoryConfig,
    faster_whisper_api_model_config_factory,
)
from faster_whisper_api.config.faster_whisper_transcription_config import (
    FasterWhisperAPITranscriptionConfig,
    faster_whisper_api_transcription_config_factory,
)


@dataclass(frozen=True, slots=True, init=False)
class AppMetaData:
    application: Final[str] = "faster-whisper-api"
    full_name: Final[str] = "faster-whisper-api:backend"
    version: Final[str] = version("faster-whisper-api")


# class Container(containers.DeclarativeContainer):

#     wiring_config = containers.WiringConfiguration(modules=[".endpoints"])

#     app_metadata: AppMetaData = providers.Configuration(default=AppMetaData())

#     model_config: FasterWhisperAPIModelFactoryConfig = providers.Configuration(pydantic_settings=faster_whisper_api_model_config_factory())
#     transription_config: FasterWhisperAPITranscriptionConfig = providers.Configuration(pydantic_settings=faster_whisper_api_transcription_config_factory())

#     giphy_client = providers.Factory(
#         giphy.GiphyClient,
#         api_key=config.giphy.api_key,
#         timeout=config.giphy.request_timeout,
#     )

#     search_service = providers.Factory(
#         services.SearchService,
#         giphy_client=giphy_client,
#     )
