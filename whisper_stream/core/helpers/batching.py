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
import math
import numpy as np
from typing import Any, Callable

from whisper_stream.core.helpers.parallel import Parallel, DEFAULT_PARALLEL_BACKEND
from whisper_stream.core.helpers.preprocessing import (
    preprocess_inputs_with_ffmpeg,
    resample_inputs_with_fn,
    DEFAULT_RESAMPLER_FN,
)
from whisper_stream.core.logger import BoundLogger
from whisper_stream.typings import (
    AudioFilesDType,
    BatchPreProcessorTasksMapping,
    BatchedTaskProcessingStrategy,
    ChunkedInputsGenerator,
    ChunkedIterationsGenerator,
    ChunkingSequenceGenerator,
    TasksCallableMapping,
    VecSignedInt,
)


def preprocess_batch_adaptive(
    inputs: list[bytes] | bytes,
    batch_size: int,
    tasks_callable_partial_mapping: TasksCallableMapping,
    chunk_length_s: float = 30.0,
    stride_length_s: float | None = None,
    audio_sampling_rate: int = 16000,
    target_sampling_rate: int = 16000,
    strategy: BatchedTaskProcessingStrategy = "smallest",
    audio_preprocessor: Callable[
        ..., tuple[AudioFilesDType, int]
    ] = preprocess_inputs_with_ffmpeg,
    fallback_resampler: Callable[
        ..., tuple[AudioFilesDType, float]
    ] = resample_inputs_with_fn,
    fallback_resampler_backend: Callable[..., AudioFilesDType] = DEFAULT_RESAMPLER_FN,
    parallel_backend: Parallel = DEFAULT_PARALLEL_BACKEND,
    logger: BoundLogger | None = None,
) -> ChunkedInputsGenerator:
    _inputs, _actual_sampling_rate = audio_preprocessor(
        data=[inputs] if isinstance(inputs, bytes) else inputs,
        sampling_rate=audio_sampling_rate,
        parallel_backend=parallel_backend,
        logger=logger,
    )

    if _actual_sampling_rate != target_sampling_rate:
        _inputs, ratio = fallback_resampler(
            data=_inputs,
            resampler_fn=fallback_resampler_backend,
            in_sampling_rate=_actual_sampling_rate,
            out_sampling_rate=target_sampling_rate,
            logger=logger,
        )
    else:
        ratio = 1.0

    asc_sorted_vector_lengths = np.array([len(vec) for vec in _inputs])
    asc_sorted_inputs_indices = np.argsort(asc_sorted_vector_lengths)
    asc_sorted_inputs = _inputs[asc_sorted_inputs_indices]

    asc_sorted_split_at_index: int = int(
        np.searchsorted(
            asc_sorted_vector_lengths[asc_sorted_inputs_indices],
            int(chunk_length_s * target_sampling_rate),
            side="right",
        )
    )

    unchunkable_inputs: AudioFilesDType = asc_sorted_inputs[:asc_sorted_split_at_index]
    chunkable_inputs: AudioFilesDType = asc_sorted_inputs[asc_sorted_split_at_index:]

    if logger is not None:
        logger.debug(
            "feature_extractor(prepocess_batch)",
            inputs_shapes=[inputs.shape for inputs in _inputs],
            unchunkable_inputs=[
                unchunkable_input.shape for unchunkable_input in unchunkable_inputs
            ],
            chunkable_inputs=[
                chunkable_input.shape for chunkable_input in chunkable_inputs
            ],
            target_sampling_rate=target_sampling_rate,
            ratio=ratio,
        )

    task_order: list[BatchedTaskProcessingStrategy] = (
        ["smallest", "largest"] if strategy == "smallest" else ["largest", "smallest"]
    )

    tasks_map: dict[BatchedTaskProcessingStrategy, BatchPreProcessorTasksMapping] = {
        "smallest": BatchPreProcessorTasksMapping(
            task_callable=tasks_callable_partial_mapping["smallest"].task_callable,
            task_kwargs={
                "inputs": unchunkable_inputs,
                "batch_size": batch_size,
                "target_sampling_rate": target_sampling_rate,
                **tasks_callable_partial_mapping["smallest"].task_kwargs,
            },
        ),
        "largest": BatchPreProcessorTasksMapping(
            task_callable=tasks_callable_partial_mapping["largest"].task_callable,
            task_kwargs={
                "inputs": chunkable_inputs,
                "chunk_length_s": chunk_length_s,
                "stride_length_s": stride_length_s,
                "batch_size": batch_size,
                "target_sampling_rate": target_sampling_rate,
                "ratio": ratio,
                **tasks_callable_partial_mapping["largest"].task_kwargs,
            },
        ),
    }

    for task in task_order:
        if len(tasks_map[task].task_kwargs["inputs"]) > 0:
            task_callable = tasks_map[task].task_callable
            task_kwargs = tasks_map[task].task_kwargs
            yield from task_callable(**task_kwargs)


def split_array_on_primary_axis(
    arr: np.ndarray[Any, Any], batch_size: int, logger: BoundLogger | None = None
) -> list[np.ndarray[Any, Any]]:
    num_batches: int = math.ceil(len(arr) / batch_size)
    batch_length: int = len(arr) // num_batches
    splits: list[np.ndarray[Any, Any]] = []
    for i in range(0, num_batches, batch_length):
        splits.append(np.array([*arr[i : i + batch_length]]))
    if logger is not None:
        logger.debug(
            "feature_extractor(preprocess_batches_for_unchunkable):split_array_on_primary_axis",
            unchunkable_inputs=[subarr.shape for subarr in arr],
            batch_size=batch_size,
            splits=[split.shape for split in splits],
        )
    return splits


def pregenerate_batching_info_for_chunkable_audio(
    chunkable_inputs: AudioFilesDType,
    chunk_length_s: float,
    stride_length_s: float | None,
    batch_size: int,
    target_sampling_rate: int,
    ratio: float,
    logger: BoundLogger | None = None,
) -> ChunkingSequenceGenerator:
    for idx, _input in enumerate(chunkable_inputs):
        if len(_input.shape) != 1:
            msg = "We expect a single channel audio input for AutomaticSpeechRecognitionPipeline"
            raise ValueError(msg)

        if chunk_length_s and _input.shape[0] / target_sampling_rate > chunk_length_s:
            # We need to chunk, as input is larger than chunk_length_s
            if stride_length_s is None:
                stride_length_s = float(chunk_length_s / 6)

            stride_length_pairs: list[float] = [
                int(round(stride_length_s * ratio)),
                int(round(stride_length_s * ratio)),
            ]

            if stride_length_pairs[0] + stride_length_pairs[1] > _input.shape[0]:
                msg = f"Stride is too large for input {idx}"
                raise ValueError(msg)

            chunk_len: int = round(chunk_length_s * target_sampling_rate)
            stride_left: int = round(stride_length_pairs[0] * target_sampling_rate)
            stride_right: int = round(stride_length_pairs[1] * target_sampling_rate)

            if chunk_len < stride_left + stride_right:
                msg = "Chunk length must be superior to stride length"
                raise ValueError(msg)

            if logger is not None:
                logger.debug(
                    "feature_extractor(preprocess_batches_for_chunkable):dispatch",
                    batch=f"#{idx}/{len(chunkable_inputs)}",
                    chunk_len=chunk_len,
                    stride_left=stride_left,
                    stride_right=stride_right,
                    batch_size=batch_size,
                    ratio=ratio,
                )

            yield (_input, chunk_len, stride_left, stride_right, batch_size)


def iterate_batch_for_chunkable_audio(
    inputs: AudioFilesDType,
    chunk_len: int,
    stride_left: int,
    stride_right: int,
    batch_size: int,
    logger: BoundLogger | None = None,
) -> ChunkedIterationsGenerator:
    inputs_len: int = inputs.shape[
        0
    ]  # Size of 1-D input / length of 1-channel audio array
    step: int = chunk_len - stride_left - stride_right

    all_chunks_start_idx: VecSignedInt = np.arange(0, inputs_len, step)
    num_samples: int = len(all_chunks_start_idx)

    num_batches: int = math.ceil(num_samples / batch_size)
    batch_idx: list[VecSignedInt] = np.array_split(np.arange(num_samples), num_batches)

    if logger is not None:
        logger.debug(
            "feature_extractor(iterate_batch_for_chunkable):gather",
            shape=inputs.shape[0],
            num_batches=num_batches,
            chunk_len=chunk_len,
            stride_left=stride_left,
            stride_right=stride_right,
            batch_size=batch_size,
        )

    for batch_counter, batch_segments in enumerate(batch_idx):
        start: float = time()
        chunk_start_idx: VecSignedInt = all_chunks_start_idx[batch_segments]

        chunk_end_idx: VecSignedInt = chunk_start_idx + chunk_len

        chunks: list[AudioFilesDType] = [
            inputs[chunk_start:chunk_end]
            for chunk_start, chunk_end in zip(chunk_start_idx, chunk_end_idx)
        ]

        _stride_left: VecSignedInt = np.where(chunk_start_idx == 0, 0, stride_left)
        is_last: VecSignedInt = np.where(
            stride_right > 0, chunk_end_idx > inputs_len, chunk_end_idx >= inputs_len
        )
        _stride_right: VecSignedInt = np.where(is_last, 0, stride_right)

        chunk_lens: list[int] = [chunk.shape[0] for chunk in chunks]
        strides: list[tuple[int, int, int]] = [
            (chunk_l, _stride_l, _stride_r)
            for chunk_l, _stride_l, _stride_r in zip(
                chunk_lens, _stride_left, _stride_right
            )
        ]

        iteration_input: dict[str, list[Any] | bool] = {
            "strides": strides,
            "chunks": chunks,
            "terminal": True if batch_counter + 1 == len(batch_idx) else False,
        }

        if logger is not None:
            logger.debug(
                "iterate_batch_for_chunkable:dispatch",
                num_chunks=len(chunk_lens),
                batch_count=f"#:{batch_counter}/{num_batches}",
                time_taken=f"{time()-start:.2}s",
                stride_left=stride_left,
                strides=strides,
                shape=[chunk.shape for chunk in chunks],
                keys=iteration_input.keys(),
            )

        yield iteration_input


__all__: list[str] = [
    "preprocess_batch_adaptive",
    "split_array_on_primary_axis",
    "pregenerate_batching_info_for_chunkable_audio",
    "iterate_batch_for_chunkable_audio",
    "BatchedTaskProcessingStrategy",
    "BatchPreProcessorTasksMapping",
]
