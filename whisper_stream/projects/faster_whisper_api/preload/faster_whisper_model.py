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
import io
from time import time
from importlib.resources import read_binary
from typing import BinaryIO, Iterable

from dependency_injector.wiring import Provide, inject

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

from whisper_stream.projects.faster_whisper_api.logger import APIBoundLogger
from whisper_stream.projects.faster_whisper_api.di.common import CommonServiceContainer
from whisper_stream.projects.faster_whisper_api.config.faster_whisper_model_config import (
    FasterWhisperAPIModelFactoryConfig,
)
from whisper_stream.projects.faster_whisper_api.config.faster_whisper_transcription_config import (
    FasterWhisperAPITranscriptionConfig,
)

from dependency_injector.wiring import Provide, inject


class AsyncWhisperModel(WhisperModel):  # type: ignore[misc]
    async def transcribe_async(
        self: WhisperModel,
        audio: BinaryIO,
        transcription_config: FasterWhisperAPITranscriptionConfig,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]:
        return self.transcribe(audio=audio, **transcription_config.model_dump())  # type: ignore[no-any-return]


@inject
async def faster_whisper_api_model_provider(
    model_logger: APIBoundLogger = Provide[CommonServiceContainer.logger],
    model_config: FasterWhisperAPIModelFactoryConfig = Provide[
        CommonServiceContainer.model_config
    ],
    transcription_config: FasterWhisperAPITranscriptionConfig = Provide[
        CommonServiceContainer.transcription_config
    ],
) -> AsyncWhisperModel:
    model = AsyncWhisperModel(**model_config.model_dump())
    start: float = time()
    model_logger.info(f"Model loaded in: {time() - start:.2f}s")
    audio = io.BytesIO(
        read_binary(
            "whisper_stream.projects.faster_whisper_api.preload.data", "audio.mp3"
        )
    )
    segment, info = model.transcribe(audio=audio, **transcription_config.model_dump())
    model_logger.info(
        f"Warm up request completed in: {time() - start:.2f}s",
        segment=segment,
        info=info,
    )
    return model


__all__: list[str] = ["faster_whisper_api_model_provider", "AsyncWhisperModel"]
