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
from typing import Any, Iterable, Literal, TypeGuard

from whisper_stream.projects.faster_whisper_api.config.base import APIBaseSettings
from pydantic import Field, ValidationError, validator


class FasterWhisperAPITransciptionVADConfig(APIBaseSettings):
    threshold: float = Field(default=0.5, gt=1e-2, le=0.99)
    min_speech_duration_ms: int = Field(default=250, ge=50, multiple_of=50)
    max_speech_duration_s: float = Field(default=float(1800))
    min_silence_duration_ms: int = Field(default=1500, multiple_of=100, ge=250)
    window_size_samples: Literal[512, 1024, 1536, 1536] = Field(default=1024)
    speech_pad_ms: int = Field(default=400, multiple_of=100, ge=100)


class FasterWhisperAPITranscriptionConfig(APIBaseSettings):
    beam_size: int = Field(default=5, ge=1, le=5)
    language: str = Field(default="en")
    task: str = Field(default="transcribe")
    best_of: int = Field(default=5, gt=1, le=5)
    patience: float = Field(default=1.0, ge=1e-1, le=1.0)
    length_penalty: float = Field(default=1.0, ge=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    no_repeat_ngram_size: int = Field(default=0, ge=0)
    log_prob_threshold: float | None = Field(default=-1.0, lt=0.0)
    no_speech_threshold: float | None = Field(default=0.6, ge=1e-1)
    compression_ratio_threshold: float | None = Field(default=2.4)
    condition_on_previous_text: bool = Field(default=True)
    prompt_reset_on_temperature: float = Field(default=0.5, gt=0.0)
    temperature: list[float] = Field(
        default=[
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ]
    )
    initial_prompt: str | Iterable[int] | None = None
    prefix: str | None = None
    suppress_blank: bool = Field(default=True)
    suppress_tokens: list[int] | None = Field(default=[-1])
    without_timestamps: bool = Field(default=False)
    max_initial_timestamp: float = Field(default=1.0, ge=0.02)
    word_timestamps: bool = Field(default=False)
    prepend_punctuations: str = Field(default="\"'“¿([{-", min_length=1)
    append_punctuations: str = Field(default="\"'.。,，!！?？:：”)]}、", min_length=1)
    vad_filter: bool = Field(default=False, validation_alias="use_vad")
    vad_parameters: FasterWhisperAPITransciptionVADConfig = (
        FasterWhisperAPITransciptionVADConfig()
    )

    @validator("temperature")
    def validate_temperature(cls, v: Any) -> list[float]:
        def typeguard_list_float(v: Any) -> TypeGuard[list[float]]:
            if not all(isinstance(x, (float, int)) for x in v):
                return False
            return True

        min_value = 0.0
        max_value = 1.0
        incorrect_container_type_error: str = (
            f"`temperature` value must be a `list` type, got {type(v)}"
        )
        incorrect_subtypes_error: str = (
            f"expected all `temperature` values to be of type `float`"
        )
        out_of_bounds_error: str = (
            f"All values in `temperature` must be between {min_value} and {max_value}"
        )
        if not isinstance(v, list):
            raise ValidationError(incorrect_container_type_error)
        if not typeguard_list_float(v):
            raise ValidationError(incorrect_subtypes_error)
        if not all(min_value <= x <= max_value for x in v):
            raise ValidationError(out_of_bounds_error)
        return v

    @validator("suppress_tokens")
    def validate_suppress_tokens(cls, v: Any | None) -> list[int] | None:
        def typeguard_list_int(v: Any) -> TypeGuard[list[int]]:
            if not all(isinstance(x, int) for x in v):
                return False
            return True

        incorrect_container_type_error: str = (
            f"`suppress_tokens` value must be a `list` type, got {type(v)}"
        )
        incorrect_subtypes_error: str = (
            f"expected all `suppress_tokens` values to be of type `int`"
        )
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValidationError(incorrect_container_type_error)
        if not typeguard_list_int(v):
            raise ValidationError(incorrect_subtypes_error)
        return v


__all__: list[str] = [
    "FasterWhisperAPITranscriptionConfig",
    "FasterWhisperAPITransciptionVADConfig",
]
