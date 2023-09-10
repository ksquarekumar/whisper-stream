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
from typing import Literal
from faster_whisper.transcribe import Segment
from whisper_stream.projects.faster_whisper_api.config.base import APIBaseModel
from whisper_stream.projects.faster_whisper_api.config.faster_whisper_model_config import (
    FasterWhisperAPIModelFactoryConfig,
)
from whisper_stream.projects.faster_whisper_api.config.faster_whisper_transcription_config import (
    FasterWhisperAPITranscriptionConfig,
)


class FasterWhisperAPIAudioTranscriptionResponseInfo(APIBaseModel):
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: list[tuple[str, float]] | None = []


class FasterWhisperAPIAudioTranscriptionResponseMetadata(APIBaseModel):
    time_taken: str
    transciption_metadata: FasterWhisperAPIAudioTranscriptionResponseInfo
    whisper_model_config: FasterWhisperAPIModelFactoryConfig
    transcription_config: FasterWhisperAPITranscriptionConfig


class FasterWhisperAPIAudioTranscriptionResponse(APIBaseModel):
    text_response: str
    segments: list[Segment] | Literal["Disabled"] = "Disabled"
    metadata: FasterWhisperAPIAudioTranscriptionResponseMetadata


__all__: list[str] = [
    "FasterWhisperAPIAudioTranscriptionResponse",
    "FasterWhisperAPIAudioTranscriptionResponseMetadata",
    "FasterWhisperAPIAudioTranscriptionResponseInfo",
]
