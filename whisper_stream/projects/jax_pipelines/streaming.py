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
import time
from typing import Any, Callable, Final, Generator, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.core.frozen_dict import freeze
from flax.training.common_utils import shard
from jax.experimental.compilation_cache import compilation_cache as cc

from flax.core import FrozenDict

from jax.sharding import PartitionSpec as P
from jax._src.stages import Wrapped
from joblib import Parallel
from transformers import WhisperProcessor
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput

from transformers.generation.configuration_utils import GenerationConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from whisper_stream.core.constants import WhisperValidCheckpoints, WhisperValidTasks
from whisper_stream.core.helpers.batching import (
    BatchPreProcessorTasksMapping,
    BatchedTaskProcessingStrategy,
    iterate_batch_for_chunkable_audio,
    pregenerate_batching_info_for_chunkable_audio,
    preprocess_batch_adaptive,
    split_array_on_primary_axis,
)
from whisper_stream.core.helpers.parsing import LanguageIDs, is_bytes, is_bytes_array
from whisper_stream.core.helpers.preprocessing import (
    DEFAULT_RESAMPLER_FN,
    preprocess_inputs_with_ffmpeg,
    resample_inputs_with_fn,
)

from whisper_stream.projects.jax_pipelines.constants import (
    JAX_CACHE_PATH,
    JAXScalarDType,
    Parallel,
    delayed,
    JAXThreadParallel,
    JAXValidDtypesMapping,
    WhisperFeatureExtractor,
    JAXWhisperTokenizers,
    JAXDecoderIDType,
    JaxShardingRulesetType,
    FLAX_DEFAULT_DP_LOGICAL_AXES,
)
from whisper_stream.core.logger import (
    LogLevelNames,
    BoundLogger,
    get_application_logger,
)
from whisper_stream.core.typings import (
    AudioFilesDType,
    ChunkedInputsGenerator,
    NDArrayFloat,
    TasksCallableMapping,
)
from whisper_stream.vendored.whisper_jax import (
    FlaxWhisperForConditionalGeneration,
    PjitPartitioner,
    PjittedFnWithContext,
    InferenceState,
)


class JAXStreamingPipeline:
    num_devices: int
    device_type: Literal["GPU", "CPU"]
    device_class: list[str]
    min_batch_size: int
    model: FlaxWhisperForConditionalGeneration
    params: FrozenDict[Any, Any]
    p_generate: PjittedFnWithContext | Wrapped

    def __init__(
        self,
        checkpoint: WhisperValidCheckpoints = "openai/whisper-small",
        dtype: JAXScalarDType = JAXValidDtypesMapping["BFLOAT16"],
        batch_size: int | None = None,
        max_length: int | None = None,
        min_log_level: LogLevelNames | None = "INFO",
    ) -> None:
        """
        Args:
            checkpoint (WhisperValidCheckpoints, optional):
                The Whisper checkpoint to use in the pipeline, Defaults to "openai/whisper-small".
                Must be an available checkpoint on the Hugging Face Hubwith Flax weights.
            dtype (JAXScalarDType, optional):
                The data type of the computation, Defaults to JAXValidDtypesMapping["BFLOAT16"].
                Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
                This can be used to enable half-precision inference on GPUs or TPUs.
            batch_size (int, optional):
                The batch size to be used in chunking transcription, Defaults to 1.
                batch size in the `__init__` method will be superseded by any batch size passed to the `__call__` method.
            max_length (int | None, optional):
                The maximum numbers of tokens to generate. Defaults to `model.config.max_length`.
        """
        self._get_devices()
        self.checkpoint: Final[WhisperValidCheckpoints] = checkpoint
        self.dtype: JAXScalarDType = dtype
        self.logger: Final[BoundLogger] = get_application_logger(
            scope="pipeline", min_log_level=min_log_level
        )

        self.initialized: bool = False
        self.is_sharded: bool = False

        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(
            self.checkpoint
        )
        self.feature_extractor: WhisperFeatureExtractor = (
            self.processor.feature_extractor
        )
        self.tokenizer: JAXWhisperTokenizers = self.processor.tokenizer

        self.model, self.params = FlaxWhisperForConditionalGeneration.from_pretrained(
            self.checkpoint,
            _do_init=False,
            dtype=self.dtype,
        )

        self.max_length: int = (
            max_length
            if max_length is not None
            else self.model.generation_config.max_length
        )
        self.model_outputs: list[dict[str, Any]] = []
        self.batch_size = (
            batch_size if batch_size is not None else self.min_batch_size
        )  # atleast 1 batch per device

        def generate(
            params: dict[Any, Any],
            input_features: NDArrayFloat,
            forced_decoder_ids: JAXDecoderIDType,
            return_timestamps: bool,
        ) -> FlaxBeamSearchOutput | FlaxGreedySearchOutput:
            output_ids: FlaxBeamSearchOutput | FlaxGreedySearchOutput = (
                self.model.pipeline_generate(
                    input_features,
                    params=params,
                    forced_decoder_ids=forced_decoder_ids,
                    return_timestamps=return_timestamps,
                    max_length=self.max_length,
                )
            )
            return output_ids

        # use pmap for DP by default if we are not on CPU
        if self.device_type != "CPU":
            self.params = jax_utils.replicate(self.params)  # type: ignore[no-untyped-call]
            self.p_generate = jax.pmap(
                generate,
                "input_features",
                in_axes=(0, 0, None),
                out_axes=0,
                static_broadcasted_argnums=(3,),
            )
        else:
            self.p_generate = jax.jit(generate)

        self.logger_binds: dict[str, str | list[str]] = {
            "model": str(checkpoint),
            "checkpoint": self.checkpoint,
            "dtype": str(self.dtype),
            "batch_size": str(self.batch_size),
            "devices": self.device_classes,
        }
        self.logger.bind(**self.logger_binds)

    def _get_devices(self) -> None:
        devices: list[jax.Device] = jax.devices()
        self.num_devices = len(devices)
        self.device_type = "CPU" if jax.devices()[0].device_kind == "cpu" else "GPU"
        self.device_classes = list(str(x.device_kind) for x in jax.devices())
        self.min_batch_size: int = jax.local_device_count()

    def _check_is_initialized(self) -> None:
        if self.initialized != True:
            error: str = "model not yet intialized, initialize by call .initialize_pipeline first"
            raise RuntimeError(error)

    @staticmethod
    def validate_data(data: list[bytes] | bytes | Any) -> list[bytes] | bytes:  # type: ignore[misc]
        if is_bytes_array(data) or is_bytes(data):
            return data
        message: str = (
            f"data must be of type bytes or list[bytes], received {type(data)}"
        )
        raise ValueError(message)

    def shard_params(
        self,
        num_mp_partitions: int = 1,
        logical_axis_rules: JaxShardingRulesetType = FLAX_DEFAULT_DP_LOGICAL_AXES,
    ) -> None:
        # do not run on CPU
        if self.device_type == "CPU":
            self.logger.debug(
                f"`.shard` is not meant to be called on {self.device_type} of class: {self.device_class}, skipping..."
            )
            return None

        def init_fn() -> FrozenDict[Any, Any]:
            input_shape = (1, 80, 3000)

            input_features = jnp.zeros(input_shape, dtype="f4")
            input_features = input_features.at[(..., -1)].set(
                self.model.config.eos_token_id
            )

            decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

            rng = jax.random.PRNGKey(0)
            init_params: FrozenDict[Any, Any] = self.model.module.init(
                rng,
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_position_ids=decoder_position_ids,
                return_dict=False,
            )
            return init_params

        # Axis names metadata
        param_axes = jax.eval_shape(init_fn)["params_axes"]

        # Create InferenceState, since the partitioner expects it
        state = InferenceState(
            step=jnp.array(0),
            params=freeze(self.model.params_shape_tree),
            params_axes=freeze(param_axes),
            flax_mutables=None,
            flax_mutables_axes=param_axes,
        )

        partitioner = PjitPartitioner(
            num_partitions=num_mp_partitions, logical_axis_rules=logical_axis_rules
        )

        mesh_axes = partitioner.get_mesh_axes(state)
        params_spec = mesh_axes.params

        p_shard_params: PjittedFnWithContext = partitioner.partition(
            self.model.to_bf16, (params_spec,), params_spec
        )

        # This will auto-magically run in mesh context
        self.params = p_shard_params(freeze(jax_utils.unreplicate(self.params)))  # type: ignore[no-untyped-call]
        self.is_sharded = True

        def generate(
            params: dict[Any, Any],
            input_features: NDArrayFloat,
            forced_decoder_ids: JAXDecoderIDType,
            return_timestamps: bool,
        ) -> FlaxBeamSearchOutput | FlaxGreedySearchOutput:
            output_ids: FlaxBeamSearchOutput | FlaxGreedySearchOutput = (
                self.model.pipeline_generate(
                    input_features,
                    params=params,
                    forced_decoder_ids=forced_decoder_ids,
                    return_timestamps=return_timestamps,
                    max_length=self.max_length,
                )
            )
            return output_ids

        # Use pjit for generate only once we've sharded the params
        self.p_generate = partitioner.partition(
            generate,
            in_axis_resources=(params_spec, P("data"), None),  # type: ignore[no-untyped-call]
            out_axis_resources=P("data"),  # type: ignore[no-untyped-call]
            static_argnums=(3,),
        )

    def generate(
        self,
        input_features: NDArrayFloat,
        language: str | None = None,
        task: WhisperValidTasks | None = None,
        return_timestamps: bool = False,
    ) -> FlaxBeamSearchOutput | FlaxGreedySearchOutput:
        forced_decoder_ids = self.get_forced_decoder_ids(
            language=language, task=task, return_timestamps=return_timestamps
        )
        if not self.is_sharded:
            # if we're using pmap we need to manually replicate the input data across devices and gather the output tokens
            output_ids: FlaxBeamSearchOutput | FlaxGreedySearchOutput = self.p_generate(
                freeze(self.params), shard(input_features), forced_decoder_ids, return_timestamps  # type: ignore[no-untyped-call]
            ).sequences
            output_ids = jax.device_get(output_ids.reshape(-1, self.max_length))
        else:
            # pjit handles replication / gathering for us auto-magically
            output_ids = self.p_generate(
                freeze(self.params),
                input_features,
                forced_decoder_ids,
                return_timestamps,
            ).sequences
        return output_ids

    def get_forced_decoder_ids(
        self,
        generation_config: GenerationConfig | None = None,
        task: WhisperValidTasks | None = None,
        language: str | LanguageIDs | None = None,
        return_timestamps: bool = False,
    ) -> JAXDecoderIDType:
        _generation_config: GenerationConfig = (
            self.model.generation_config
            if generation_config is None
            else generation_config
        )

        if hasattr(_generation_config, "is_multilingual"):
            is_multilingual = _generation_config.is_multilingual
        else:
            is_multilingual = None

        forced_decoder_ids: JAXDecoderIDType = []

        if is_multilingual:
            if language is not None:
                language = str(language).lower()
                if language in _generation_config.lang_to_id.keys():
                    language_token: str = language
                elif language in TO_LANGUAGE_CODE.values():
                    language_token = f"<|{language}|>"
                elif language in TO_LANGUAGE_CODE.keys():
                    language_token = f"<|{TO_LANGUAGE_CODE[language]}|>"
                else:
                    if len(language) == 2:
                        # ISO 639-1 language code
                        acceptable_languages: list[str] = list(
                            TO_LANGUAGE_CODE.values()
                        )
                    elif "<" in language or "|" in language or ">" in language:
                        # generation config language code
                        acceptable_languages = list(
                            _generation_config.lang_to_id.keys()
                        )
                    else:
                        # language passed as a string
                        acceptable_languages = list(TO_LANGUAGE_CODE.keys())
                    msg = f"Unsupported language: {language}. Language should be one of: {acceptable_languages}."
                    raise ValueError(msg)
                forced_decoder_ids.append(
                    (1, _generation_config.lang_to_id[language_token])
                )

            if task is not None:
                forced_decoder_ids.append((2, _generation_config.task_to_id[task]))
            else:
                forced_decoder_ids.append(
                    (2, _generation_config.task_to_id["transcribe"])
                )

        if not return_timestamps:
            if (
                forced_decoder_ids
                and forced_decoder_ids[-1][0]
                != _generation_config.no_timestamps_token_id
            ):
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append(
                    (idx, _generation_config.no_timestamps_token_id)
                )

        return forced_decoder_ids

    def _iterate_batch_for_chunkable(
        self,
        single_input: AudioFilesDType,
        chunk_len: int,
        stride_left: int,
        stride_right: int,
        batch_size: int,
        do_normalize: bool = True,
    ) -> ChunkedInputsGenerator:
        for batch_counter, iteration_input in enumerate(
            iterate_batch_for_chunkable_audio(
                single_input=single_input,
                chunk_len=chunk_len,
                stride_left=stride_left,
                stride_right=stride_right,
                batch_size=batch_size,
                logger=self.logger,
            )
        ):
            processed: BatchFeature = self.feature_extractor(
                raw_speech=iteration_input.get("chunks"),
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="np",
                do_normalize=do_normalize,
            )

            iteration_input_processed = {
                "terminal": iteration_input["terminal"],
                "strides": iteration_input["strides"],
                **processed,
            }

            self.logger.debug(
                "feature_extractor(iterate_batch_for_chunkable):dispatch",
                batch_counter=batch_counter,
                shape=processed["input_features"].shape,
                keys=iteration_input_processed.keys(),
            )

            yield iteration_input_processed

    def _generate_batching_info_for_chunkable(
        self,
        inputs: list[AudioFilesDType],
        chunk_length_s: float,
        stride_length_s: float | None,
        batch_size: int,
        target_sampling_rate: int,
        ratio: float,
        do_normalize: bool = True,
    ) -> ChunkedInputsGenerator:
        for batch_input in pregenerate_batching_info_for_chunkable_audio(
            chunkable_inputs=inputs,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            target_sampling_rate=target_sampling_rate,
            ratio=ratio,
            logger=self.logger,
        ):
            (
                _single_input,
                _chunk_len,
                _stride_left,
                _stride_right,
                _batch_size,
            ) = batch_input

            yield from self._iterate_batch_for_chunkable(
                single_input=_single_input,
                chunk_len=_chunk_len,
                stride_left=_stride_left,
                stride_right=_stride_right,
                batch_size=_batch_size,
                do_normalize=do_normalize,
            )

    def _preprocess_batches_for_unchunkable(
        self,
        inputs: list[AudioFilesDType],
        batch_size: int,
        target_sampling_rate: int,
        do_normalize: bool = True,
    ) -> ChunkedInputsGenerator:
        unchunkable_inputs: list[AudioFilesDType] = inputs
        batches: list[list[AudioFilesDType]] = (
            split_array_on_primary_axis(unchunkable_inputs, batch_size)
            if len(unchunkable_inputs) > 1
            else [inputs]
        )

        self.logger.debug(
            "feature_extractor(preprocess_batches_for_unchunkable):gather",
            unchunkable_inputs=[
                unchunkable_input.shape for unchunkable_input in unchunkable_inputs
            ],
            num_files=len(unchunkable_inputs),
            batch_size=batch_size,
            num_batches=len(batches),
            batches=[f"len={len(batch)}" for batch in batches],
            target_sampling_rate=target_sampling_rate,
        )

        for idx, _batched_input in enumerate(batches):
            start: float = time.time()
            processed: BatchFeature = self.feature_extractor(
                raw_speech=_batched_input,
                sampling_rate=target_sampling_rate,
                return_tensors="np",
                do_normalize=do_normalize,
            )
            _processed: dict[str, Any] = {"fused_inputs": True, **processed}
            self.logger.debug(
                "feature_extractor(preprocess_batches_for_unchunkable):dispatch",
                size=len(_batched_input),
                time_taken=f"{time.time()-start:.2}s",
                shape=processed["input_features"].shape,
                keys=processed.keys(),
                iteration=f"{idx}/{len(batches)}",
            )
            yield _processed

    def preprocess_batch(
        self,
        inputs: list[bytes] | bytes,
        chunk_length_s: float = 30.0,
        stride_length_s: float | None = None,
        batch_size: int | None = None,
        sampling_rate: int = 16000,
        strategy: BatchedTaskProcessingStrategy = "smallest",
        do_normalize: bool = True,
        parallel_backend: Parallel = JAXThreadParallel,
    ) -> ChunkedInputsGenerator:
        # runtime
        self.validate_data(inputs)

        _batch_size: int = batch_size if batch_size is not None else self.batch_size

        tasks_callable_partial_mapping: TasksCallableMapping = {
            "smallest": BatchPreProcessorTasksMapping(
                task_callable=self._preprocess_batches_for_unchunkable,
                task_kwargs={"do_normalize": do_normalize},
            ),
            "largest": BatchPreProcessorTasksMapping(
                task_callable=self._generate_batching_info_for_chunkable,
                task_kwargs={"do_normalize": do_normalize},
            ),
        }

        target_sampling_rate: int = self.feature_extractor.sampling_rate

        yield from preprocess_batch_adaptive(
            inputs=inputs,
            batch_size=_batch_size,
            tasks_callable_partial_mapping=tasks_callable_partial_mapping,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            audio_sampling_rate=sampling_rate,
            target_sampling_rate=target_sampling_rate,
            strategy=strategy,
            audio_preprocessor=self.preprocess_inputs_with_ffmpeg,
            fallback_resampler=self.resample_inputs_with_fn,
            fallback_resampler_backend=DEFAULT_RESAMPLER_FN,
            parallel_backend=parallel_backend,
            logger=self.logger,
        )

    def postprocess(
        self,
        model_outputs: list[dict[Any, list[Any]]],
        return_timestamps: bool | None = None,
        return_language: str | None = None,
        fused_outputs: bool = False,
        parallel_backend: Parallel = JAXThreadParallel,
    ) -> list[dict[str, Any]]:
        # unpack the outputs from list(dict(list)) to list(dict)
        _model_outputs: list[dict[Any, Any]] = [
            dict(zip(output, t))
            for output in model_outputs
            for t in zip(*output.values())
        ]

        time_precision: float = (
            self.feature_extractor.chunk_length / self.model.config.max_source_positions
        )
        # Send the chunking back to seconds, it's easier to handle in whisper
        sampling_rate: int = self.feature_extractor.sampling_rate

        self.logger.debug(
            "postprocess:incoming",
            num_output_batches_received=len(model_outputs),
            num_output_segments=len(_model_outputs),
            time_precision=time_precision,
            return_timestamps=return_timestamps,
            return_language=return_language,
            sampling_rate=sampling_rate,
            fused_outputs=fused_outputs,
        )
        if fused_outputs == True:
            results: list[tuple[str, dict[Any, Any]]] = (
                list(
                    parallel_backend(
                        delayed(self.tokenizer._decode_asr)(
                            [_model_output],
                            return_timestamps=return_timestamps,
                            return_language=return_language,
                            time_precision=time_precision,
                        )
                        for _model_output in _model_outputs
                    )
                )
                if len(_model_outputs) > 1
                else [
                    self.tokenizer._decode_asr(
                        _model_outputs,
                        return_timestamps=return_timestamps,
                        return_language=return_language,
                        time_precision=time_precision,
                    )
                ]
            )
            return [{"text": result[0], **result[1]} for result in results]

        for output in _model_outputs:
            if "stride" in output:
                chunk_len, stride_left, stride_right = output["stride"]
                # Go back in seconds
                chunk_len /= sampling_rate
                stride_left /= sampling_rate
                stride_right /= sampling_rate
                output["stride"] = chunk_len, stride_left, stride_right

        text, optional = self.tokenizer._decode_asr(
            _model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )
        return [{"text": text, **optional}]

    def forward(
        self,
        model_inputs: dict[Any, Any],
        batch_size: int | None = None,
        language: str | None = None,
        task: WhisperValidTasks | None = None,
        return_timestamps: bool = False,
    ) -> dict[str, Any]:
        # We need to keep track of some additional input arguments for post-processing so need to forward these on after running generation
        input_features = model_inputs.pop("input_features")
        input_batch_size = input_features.shape[0]

        if input_batch_size != batch_size:
            padding = np.zeros(
                [batch_size - input_batch_size, *input_features.shape[1:]],
                input_features.dtype,
            )
            input_features = np.concatenate([input_features, padding])

        pred_ids = self.generate(
            input_features,
            language=language,
            task=task,
            return_timestamps=return_timestamps,
        )[:input_batch_size]

        # tokenizer's decode method expects an extra dim - we insert it here for convenience
        out = {"tokens": pred_ids[:, None, :]}

        stride = model_inputs.pop("stride", None)
        if stride is not None:
            out["stride"] = stride

        return out

    def __call__(
        self,
        inputs: bytes | list[bytes],
        chunk_length_s: float = 30.0,
        stride_length_s: float | None = None,
        batch_size: int | None = None,
        language: str | None = None,
        task: WhisperValidTasks = "transcribe",
        return_timestamps: bool = False,
        do_normalize: bool = True,
        strategy: BatchedTaskProcessingStrategy = "smallest",
    ) -> Generator[dict[str, Any] | list[dict[str, Any]], Any, None]:
        """
        Transcribe an audio input sequence to a text transcription, optionally with timestamps.

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                The inputs is either:
                    - `bytes` is the byte content of an audio file and is interpreted by *ffmpeg* in the
                      same way.
                    - `list[bytes]` multiple bytes packed together
            chunk_length_s (`float`, *optional*, defaults to 30.0):
                The input length for each chunk. If `chunk_length_s = 0` then chunking is disabled. By default, the chunk
                length is set 30.0s, equal to Whisper's context window.
            stride_length_s (`float`, *optional*, defaults to `chunk_length_s / 6`):
                The length of stride on the left and right of each chunk. Used only with `chunk_length_s > 0`. This enables
                the model to *see* more context and infer letters better than without this context but the pipeline
                discards the stride bits at the end to make the final reconstitution as perfect as possible.

                <Tip>

                For more information on how to effectively use `stride_length_s`, refer to the [ASR chunking
                blog post](https://huggingface.co/blog/asr-chunking).

                </Tip>
            batch_size (`int`, *optional*, defaults to the minimum per-device batch size, i.e. `jax.local_device_count()`):
                The batch size to be used in chunking transcription. Beneficial for transcribing long audio files. Passing
                a batch size in the `__call__` method will supersede any batch size passed to the `__init__`.
            task (`str`, *optional*):
                Task to use for generation, either `"transcribe"` or `"translate"`. Defaults to `"transcribe"`.
            language (`str`, *optional*):
                Language token to use for generation, can be either in the form of `"<|en|>"`, `"en"` or `"english"`.
                Defaults to `None`, meaning the language is automatically inferred from the audio input.
            return_timestamps (*optional*, `bool`):
                Whether to return timestamps in the prediction. Defaults to False. If set to true, the pipeline
                will return two keys in the output dictionary: `"text"` containing the text transcription, and `"chunks"`
                containing the transcription segments chunked by their utterance-level timestamps.
            do_normalize (`bool`, *optional*, defaults to `False`):
                Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
                improve the performance of the model.
            strategy(BatchedTaskProcessingStrategy):
                strategy parameter to control whether to process the smallest/largest outputs first, Defaults to "smallest",

        Yields:
            `Dict`: A dictionary with the following keys:
                - **text** (`str` ) -- The recognised text.
                - **chunks** (*optional(, `List[Dict]`)
                    When using `return_timestamps`, the `chunks` will become a list containing all the various text
                    chunks identified by the model, *e.g.* `[{"text": "hi ", "timestamps": (0.5,0.9), {"text":
                    "there", "timestamps": (1.0, 1.5)}]`. The original full text can roughly be recovered by doing
                    `"".join(chunk["text"] for chunk in output["chunks"])`.
            `List[Dict]`: A list of dictionaries with the same keys as above.
        """
        self._check_is_initialized()

        batch_size = batch_size if batch_size is not None else self.batch_size
        if batch_size % self.min_batch_size != 0:
            msg = f"Batch size must be a multiple of the number of JAX devices, but got batch size {batch_size} and num devices {self.min_batch_size}."
            raise ValueError(msg)

        dataloader = self.preprocess_batch(
            inputs,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            strategy=strategy,
            do_normalize=do_normalize,
        )
        # iterate over our chunked audio samples
        for batch in dataloader:
            output: dict[str, Any] = self.forward(
                {**batch},
                batch_size=batch_size,
                language=language,
                task=task,
                return_timestamps=return_timestamps,
            )
            if batch.get("fused_inputs") is True:
                post_processed_fused: list[dict[str, Any]] = self.postprocess(
                    model_outputs=[output],
                    return_timestamps=return_timestamps,
                    fused_outputs=True,
                )
                yield post_processed_fused
            if batch.get("terminal") is False:
                self.model_outputs.append(output)
                continue
            if batch.get("terminal") == True:
                self.model_outputs.append(output)
                post_processed: list[dict[str, Any]] = self.postprocess(
                    model_outputs=self.model_outputs,
                    return_timestamps=return_timestamps,
                )
                self.model_outputs = []
                yield post_processed

    def initialize_pipeline(
        self,
        batch_size: int | None = None,
        language: str = "english",
        task: WhisperValidTasks = "transcribe",
        return_timestamps: bool = False,
        shard_with: JaxShardingRulesetType | Literal[True] | None = None,
        use_experimental_cache: bool = True,
    ) -> None:
        """instantiate and return the Pipeline with internal batching, meant for large files that can be chunked internally.
        sets `~JAXStreamingPipeline.model` after initialising it.

        Args:
            batch_size (`int`, *optional*, defaults to the minimum per-device batch size, i.e. `jax.local_device_count()`):
                The batch size to be used in chunking transcription. Beneficial for transcribing long audio files. Passing
                a batch size in the `__call__` method will supersede any batch size passed to the `__init__`.
            language (str):
                The language to perform the task in, defaults to `english`
            task (ValidJaxTasks):
                The mode of the task, one of `transcribe` or `translate`, defaults to `transcribe`
            return_timestamps (bool):
                whether to return the timestamps of the generated text.
            use_experimental_cache (bool):
                use `experimental.compilation_cache` for compilation phase, defaults to True.
                usage will speed-up the compilation time for subsequent compilations across restarts

        Raises:
            ValueError: if neither of `sample_data: bytes` cannot be read.

        Returns:
            None
        """
        self.logger.info(f"Initializing {self.checkpoint}/{self.dtype} pipeline")

        if use_experimental_cache:
            cc.initialize_cache(JAX_CACHE_PATH)  # type: ignore[no-untyped-call]

        _batch_size = batch_size if batch_size is not None else self.batch_size
        _random_inputs = {"input_features": np.ones((_batch_size, 80, 3000))}
        self.logger.info(f"Compiling {self.checkpoint}/{self.dtype} pipeline")
        start: float = time.time()

        _shard_with = FLAX_DEFAULT_DP_LOGICAL_AXES if shard_with == True else shard_with

        if _shard_with is not None and self.device_type != "CPU":
            self.shard_params(
                num_mp_partitions=self.num_devices, logical_axis_rules=_shard_with
            )

        _: dict[str, Any] = self.forward(
            _random_inputs,
            batch_size=_batch_size,
            language=language,
            task=task,
            return_timestamps=return_timestamps,
        )
        self.initialized = True
        self.logger.info(f"Compilation done in {time.time() - start:.2f}s")

    def preprocess_inputs_with_ffmpeg(
        self,
        data: list[bytes],
        sampling_rate: int = 16000,
        parallel_backend: Parallel = JAXThreadParallel,  # Always takes a list of inputs
        logger: BoundLogger | None = None,
    ) -> tuple[list[AudioFilesDType], int]:
        # returns batch_size x VEC[data_length[float32]],for a single input batch_size will be 1
        # so returned array will have a shape of (batches, data_length)
        return preprocess_inputs_with_ffmpeg(
            data=data,
            sampling_rate=sampling_rate,
            parallel_backend=parallel_backend,
            logger=logger or self.logger,
        )

    def resample_inputs_with_fn(
        self,
        data: list[AudioFilesDType],  #
        resampler_fn: Callable[..., AudioFilesDType],
        in_sampling_rate: int,
        out_sampling_rate: int = 16000,
        parallel_backend: Parallel = JAXThreadParallel,
        logger: BoundLogger | None = None,
    ) -> tuple[list[AudioFilesDType], float]:
        # accepts batch_size x VEC[data_length[float32]]
        # returns batch_size x VEC[data_length[float32]]
        return resample_inputs_with_fn(
            data=data,
            in_sampling_rate=in_sampling_rate,
            out_sampling_rate=out_sampling_rate,
            resampler_fn=resampler_fn,
            parallel_backend=parallel_backend,
            logger=logger or self.logger,
        )
