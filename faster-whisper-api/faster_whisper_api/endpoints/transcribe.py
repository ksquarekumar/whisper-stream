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
from typing import Iterable, Literal, Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends
from faster_whisper.transcribe import Segment, TranscriptionInfo
from pydantic import ValidationError

from faster_whisper_api.config.faster_whisper_model_config import (
    FasterWhisperAPIModelFactoryConfig,
)
from faster_whisper_api.config.faster_whisper_transcription_config import (
    FasterWhisperAPITranscriptionConfig,
)
from faster_whisper_api.di.common import CommonServiceContainer
from faster_whisper_api.di.faster_whisper_model import (
    FasterWhisperModelServiceContainer,
)
from faster_whisper_api.logger import APILogger
from faster_whisper_api.preload.faster_whisper_model import AsyncWhisperModel
from faster_whisper_api.schemas.exceptions import (
    BadRequestAPIException,
    InternalServerAPIException,
    UnprocessableEntityAPIException,
)
from faster_whisper_api.schemas.responses import (
    FasterWhisperAPIAudioTranscriptionResponse,
    FasterWhisperAPIAudioTranscriptionResponseInfo,
    FasterWhisperAPIAudioTranscriptionResponseMetadata,
)

from fastapi import File

transcription_router = APIRouter()


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
    logger: APILogger = Depends(Provide[CommonServiceContainer.logger]),
) -> FasterWhisperAPIAudioTranscriptionResponse:
    result: dict[Literal["response"], FasterWhisperAPIAudioTranscriptionResponse] = {}
    try:
        start: float = time()
        model_response: tuple[
            Iterable[Segment], TranscriptionInfo
        ] = await model.transcribe_async(
            audio=BytesIO(audio), transcription_config=transcription_config
        )
        segments, info = model_response

        response = FasterWhisperAPIAudioTranscriptionResponse(
            text_response="".join(segment.text for segment in segments),
            segments="Disabled",
            metadata=FasterWhisperAPIAudioTranscriptionResponseMetadata(
                time_taken=f"{time() - start:.2f}s",
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
        result["response"] = response
        logger.info(time=f"{time()-start:.2f}s")
    except ValidationError as exception:
        return BadRequestAPIException(error=exception)
    except ValueError as exception:
        return UnprocessableEntityAPIException(error=exception)
    except Exception as unknown_exc:
        return InternalServerAPIException(unknown_exc)
    finally:
        return result["response"]


__all__: list[str] = ["transcription_router"]
