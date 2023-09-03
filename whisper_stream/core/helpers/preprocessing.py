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
from time import time
from typing import Callable
import librosa

from transformers.pipelines.audio_utils import ffmpeg_read
from whisper_stream.core.helpers.parallel import (
    Parallel,
    delayed,
    DEFAULT_PARALLEL_BACKEND,
)
from whisper_stream.core.logger import BoundLogger
from whisper_stream.typings import AudioFilesDType

try:
    import librosa
except ImportError as err:
    msg: str = (
        "To support resampling audio files, please install 'librosa' and 'soundfile'."
    )
    raise ImportError(msg) from err

DEFAULT_RESAMPLER_FN: Callable[..., AudioFilesDType] = librosa.resample


def preprocess_inputs_with_ffmpeg(
    data: list[bytes],  # Always takes a list of inputs
    sampling_rate: int = 16000,
    parallel_backend: Parallel = DEFAULT_PARALLEL_BACKEND,
    logger: BoundLogger | None = None,
) -> tuple[list[AudioFilesDType], int]:
    # returns batch_size x VEC[data_length[float32]],for a single input batch_size will be 1
    # so returned array will have a shape of (batches, data_length)
    start: float = time()
    converted: list[AudioFilesDType] = (
        list(
            parallel_backend(
                delayed(ffmpeg_read)(d, sampling_rate=sampling_rate) for d in data
            )
        )
        if len(data) > 1
        else [ffmpeg_read(data[0], sampling_rate=sampling_rate)]
    )
    if logger is not None:
        logger.info(
            f"ffmpeg conversion", num_items=len(data), time_taken=f"{time()-start:.2}s"
        )
    return converted, sampling_rate


def resample_inputs_with_fn(
    data: list[AudioFilesDType],
    in_sampling_rate: int,
    out_sampling_rate: int = 16000,
    resampler_fn: Callable[..., AudioFilesDType] = DEFAULT_RESAMPLER_FN,
    parallel_backend: Parallel = DEFAULT_PARALLEL_BACKEND,
    logger: BoundLogger | None = None,
) -> tuple[list[AudioFilesDType], float]:
    # accepts batch_size x VEC[data_length[float32]]
    # returns batch_size x VEC[data_length[float32]]
    start: float = time()
    converted: list[AudioFilesDType] = (
        list(parallel_backend(
            delayed(resampler_fn)(
                d, orig_sr=in_sampling_rate, target_sr=out_sampling_rate
            )
            for d in data
        ))
        if len(data) > 1
        else [ resampler_fn(data[0], orig_sr=in_sampling_rate, target_sr=out_sampling_rate)]
    )
    if logger is not None:
        logger.info(
            f"resampling", num_items=len(data), time_taken=f"{time()-start:.2}s"
        )
    return converted, float(in_sampling_rate) / float(out_sampling_rate)


__all__: list[str] = [
    "preprocess_inputs_with_ffmpeg",
    "resample_inputs_with_fn",
    "DEFAULT_RESAMPLER_FN",
]
