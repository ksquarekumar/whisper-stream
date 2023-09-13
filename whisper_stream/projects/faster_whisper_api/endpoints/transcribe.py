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
from io import BytesIO
from time import time
from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, File
from faster_whisper.transcribe import TranscriptionInfo
from pydantic import ValidationError

from whisper_stream.core.helpers.parallel import DEFAULT_PARALLEL_BACKEND, delayed

from whisper_stream.projects.faster_whisper_api.config.faster_whisper_model_config import (
    FasterWhisperAPIModelFactoryConfig,
)
from whisper_stream.projects.faster_whisper_api.config.faster_whisper_transcription_config import (
    FasterWhisperAPITranscriptionConfig,
)
from whisper_stream.projects.faster_whisper_api.di.common import CommonServiceContainer
from whisper_stream.projects.faster_whisper_api.di.faster_whisper_model import (
    FasterWhisperModelServiceContainer,
)
from whisper_stream.projects.faster_whisper_api.preload.faster_whisper_model import (
    AsyncWhisperModel,
)
from whisper_stream.projects.faster_whisper_api.schemas.exceptions import (
    BadRequestAPIException,
    InternalServerAPIException,
    UnprocessableEntityAPIException,
)
from whisper_stream.projects.faster_whisper_api.schemas.responses import (
    FasterWhisperAPIAudioTranscriptionResponse,
    FasterWhisperAPIAudioTranscriptionResponseInfo,
    FasterWhisperAPIAudioTranscriptionResponseMetadata,
)

transcription_router = APIRouter()


async def make_response(
    text_response: str,
    info: TranscriptionInfo,
    time_taken: str,
    transcription_config: FasterWhisperAPITranscriptionConfig,
    model_config: FasterWhisperAPIModelFactoryConfig,
) -> FasterWhisperAPIAudioTranscriptionResponse:
    return FasterWhisperAPIAudioTranscriptionResponse(
        text_response=text_response,
        segments="Disabled",
        metadata=FasterWhisperAPIAudioTranscriptionResponseMetadata(
            time_taken=time_taken,
            transciption_metadata=FasterWhisperAPIAudioTranscriptionResponseInfo(
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration,
                duration_after_vad=info.duration_after_vad,
                all_language_probs=list(*(info.all_language_probs or [])),
            ),
            whisper_model_config=model_config,
            transcription_config=transcription_config,
        ),
    )


@transcription_router.post(
    "/audio/transcribe",
    status_code=200,
    operation_id="transcribe-audio",
    responses={
        400: {"model": BadRequestAPIException},
        422: {"model": UnprocessableEntityAPIException},
        500: {"model": InternalServerAPIException},
    },
)
@inject
async def transcribe_audio(
    audio: Annotated[bytes, File()],
    model: AsyncWhisperModel = Depends(
        Provide[FasterWhisperModelServiceContainer.model]
    ),
    model_config: FasterWhisperAPIModelFactoryConfig = Depends(
        Provide[CommonServiceContainer.model_config]
    ),
    transcription_config: FasterWhisperAPITranscriptionConfig = Depends(
        Provide[CommonServiceContainer.transcription_config]
    ),
) -> FasterWhisperAPIAudioTranscriptionResponse:
    try:
        start: float = time()
        segments, info = await model.transcribe_async(
            audio=BytesIO(audio), transcription_config=transcription_config
        )
        time_taken: str = f"{time() - start:.2f}s"

        text_response: str = "".join(
            list(
                DEFAULT_PARALLEL_BACKEND(
                    delayed(str)(segment[4]) for segment in segments
                )
            )
        )

        time_taken = time_taken + f"/{time() - start:.2f}s"
        return await make_response(
            text_response=text_response,
            info=info,
            time_taken=time_taken,
            transcription_config=transcription_config,
            model_config=model_config,
        )
    except ValidationError as exception:
        raise BadRequestAPIException(error=exception)
    except ValueError as exception:
        raise UnprocessableEntityAPIException(error=exception)
    except Exception as unknown_exc:
        raise InternalServerAPIException(error=unknown_exc)


__all__: list[str] = ["transcription_router"]
