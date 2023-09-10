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
from dependency_injector import containers, providers
from whisper_stream.projects.faster_whisper_api.preload.faster_whisper_model import (
    faster_whisper_api_model_provider,
)


class FasterWhisperModelServiceContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "whisper_stream.projects.faster_whisper_api.endpoints.transcribe",
            "whisper_stream.projects.faster_whisper_api.app",
        ],
        auto_wire=True,
    )

    model = providers.ThreadSafeSingleton(faster_whisper_api_model_provider)


__all__: list[str] = ["FasterWhisperModelServiceContainer"]
