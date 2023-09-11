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
from typing import Any, Literal, TypeGuard

from pydantic import Field, ValidationError, validator

from whisper_stream.core.helpers.parallel import cpu_count
from whisper_stream.projects.faster_whisper_api.config.base import APIBaseSettings


class FasterWhisperAPIModelFactoryConfig(APIBaseSettings):
    model_size_or_path: Literal[
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large",
    ] = Field(default="large-v2", alias="checkpoint")
    device: str = Field(default="auto")
    device_index: int | list[int] = Field(default=0)
    compute_type: Literal[
        "auto",
        "int8",
        "int8_float32",
        "int8_float16",
        "int8_bfloat16",
        "int16",
        "float16",
        "bfloat16",
        "float32",
    ] = Field(default="int8")
    cpu_threads: int = Field(
        default_factory=cpu_count, validation_alias="OMP_NUM_THREADS"
    )
    num_workers: int = Field(
        default_factory=cpu_count, validation_alias="OMP_NUM_THREADS"
    )
    download_root: str | None = None
    local_files_only: bool = False

    @validator("device_index")
    def validate_device_index(cls, v: Any) -> int | list[int]:
        def typeguard_list_int(v: Any) -> TypeGuard[list[int]]:
            if not all(isinstance(x, int) for x in v):
                return False
            return True

        min_value = 0

        incorrect_container_type_error: str = (
            f"value must be a `list` type, got {type(v)}"
        )
        incorrect_subtypes_error: str = f"expected all values to be of type `float`"
        out_of_bounds_error: str = f"All values in `device_index` must be >= 0"
        if isinstance(v, int):
            if v >= 0:
                return v
            else:
                raise ValidationError(out_of_bounds_error)
        if not isinstance(v, list):
            raise ValidationError(incorrect_container_type_error)
        if not typeguard_list_int(v):
            raise ValidationError(incorrect_subtypes_error)
        if not all(min_value <= x for x in v):
            raise ValidationError(out_of_bounds_error)
        return v


__all__: list[str] = ["FasterWhisperAPIModelFactoryConfig"]
