# SPDX-FileCopyrightText: 2023-present krishnakumar <krishna.kumar@peak.ai>
#
# SPDX-License-Identifier: Apache-2.0
import math
import time
from typing import Any, Callable, Final, Generator, Literal, TypeAlias

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.core.frozen_dict import freeze
from flax.training.common_utils import shard
from jax._src.numpy.lax_numpy import _ScalarMeta
from jax.experimental.compilation_cache import compilation_cache as cc

from jax.sharding import PartitionSpec as P
from jax._src.stages import Wrapped
from joblib import Parallel
from transformers import WhisperProcessor
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput

from transformers.generation.configuration_utils import GenerationConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers.models.whisper.tokenization_whisper_fast import WhisperTokenizerFast
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

from whisper_stream.constants import JAX_CACHE_PATH
from whisper_stream.logger import BoundLogger, get_application_logger
from whisper_stream.utils.helpers import LanguageIDs, is_bytes, is_bytes_array
from whisper_stream.utils.parallel import delayed, get_backend
from whisper_stream.vendored.whisper_jax import FlaxWhisperForConditionalGeneration
from whisper_stream.vendored.whisper_jax.modeling_flax_whisper import FlaxWhisperForConditionalGeneration
from whisper_stream.vendored.whisper_jax.partitioner import PjitPartitioner, PjittedFnWithContext
from whisper_stream.vendored.whisper_jax.train_state import InferenceState


JAXCheckpoints = Literal[
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
    "openai/whisper-large-v2",
]

FrozenDict = flax.core.FrozenDict

JAXScalarDType: TypeAlias = _ScalarMeta

JAXValidDtypesMapping: Final[dict[str, JAXScalarDType]] = {
    "FLOAT32": jnp.float32,
    "BFLOAT16": jnp.bfloat16,
    "FLOAT16": jnp.float16,
}
JAXValidTasks = Literal["transcribe", "translate"]

ND_ARRAY_FLOAT = np.ndarray[tuple[int, ...], np.dtype[np.floating[Any]]]
ND_ARRAY_INT = np.ndarray[tuple[int, ...], np.dtype[np.integer[Any]]]

VEC_FLOAT = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
VEC_SIGNED_INT = np.ndarray[tuple[int], np.dtype[np.signedinteger[Any]]]

AUDIO_FILES_DTYPE = np.ndarray[tuple[int, int], np.dtype[np.float32]]

DECODER_ID_TYPE = list[tuple[int, str | Any]]

JAXThreadParallel: Parallel = get_backend("threading")
JAXMPParallel: Parallel = get_backend("loky")

WhisperTokenizers: TypeAlias = WhisperTokenizerFast | WhisperTokenizer

# 2D parameter and activation partitioning for DP
SHARDING_RULESET_TYPE = tuple[tuple[str, str | None], ...]
FLAX_DEFAULT_DP_LOGICAL_AXES: SHARDING_RULESET_TYPE = (
    ("batch", "data"),
    ("mlp", None),
    ("heads", None),
    ("vocab", None),
    ("embed", None),
    ("embed", None),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None),
)


class JAXPipeline:
    num_devices: int
    device_type: Literal["GPU", "CPU"]
    device_class: list[str]
    min_batch_size: int
    model: FlaxWhisperForConditionalGeneration
    params: FrozenDict[Any, Any]
    p_generate: PjittedFnWithContext | Wrapped

    def __init__(
        self,
        checkpoint: JAXCheckpoints = "openai/whisper-small",
        dtype: JAXScalarDType = JAXValidDtypesMapping["BFLOAT16"],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> None:
        """
        Args:
            checkpoint (JAXCheckpoints, optional):
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
        self.checkpoint: Final[JAXCheckpoints] = checkpoint
        self.dtype: JAXScalarDType = dtype
        self.logger: Final[BoundLogger] = get_application_logger(name="pipeline")

        self.initialized: bool = False
        self.is_sharded: bool = False

        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(self.checkpoint)
        self.feature_extractor: WhisperFeatureExtractor = self.processor.feature_extractor
        self.tokenizer: WhisperTokenizers = self.processor.tokenizer

        self.model, self.params = FlaxWhisperForConditionalGeneration.from_pretrained(
            self.checkpoint,
            _do_init=False,
            dtype=self.dtype,
        )

        self.max_length: int = max_length if max_length is not None else self.model.generation_config.max_length

        self.batch_size = batch_size if batch_size is not None else self.min_batch_size  # atleast 1 batch per device

        def generate(
            params: dict[Any, Any],
            input_features: ND_ARRAY_FLOAT,
            forced_decoder_ids: DECODER_ID_TYPE,
            return_timestamps: bool,
        ) -> FlaxBeamSearchOutput | FlaxGreedySearchOutput:
            output_ids: FlaxBeamSearchOutput | FlaxGreedySearchOutput = self.model.pipeline_generate(
                input_features,
                params=params,
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps=return_timestamps,
                max_length=self.max_length,
            )  # type: ignore[no-untyped-call]
            return output_ids

        # use pmap for DP by default if we are not on CPU
        if self.device_type != "CPU":
            self.params = jax_utils.replicate(self.params)  # type: ignore[no-untyped-call]
            self.p_generate = jax.pmap(
                generate, "input_features", in_axes=(0, 0, None), out_axes=0, static_broadcasted_argnums=(3,)
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
        message: str = f"data must be of type bytes or list[bytes], received {type(data)}"
        raise ValueError(message)

    def shard_params(
        self,
        num_mp_partitions: int = 1,
        logical_axis_rules: SHARDING_RULESET_TYPE = FLAX_DEFAULT_DP_LOGICAL_AXES,
    ) -> None:
        # do not run on CPU
        if self.device_type == "CPU":
            self.logger.warning(
                f"`.shard` is not meant to be called on {self.device_type} of class: {self.device_class}, skipping..."
            )
            return None

        def init_fn() -> FrozenDict[Any, Any]:
            input_shape = (1, 80, 3000)

            input_features = jnp.zeros(input_shape, dtype="f4")
            input_features = input_features.at[(..., -1)].set(self.model.config.eos_token_id)

            decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

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

        partitioner = PjitPartitioner(num_partitions=num_mp_partitions, logical_axis_rules=logical_axis_rules)

        mesh_axes = partitioner.get_mesh_axes(state)
        params_spec = mesh_axes.params

        p_shard_params: PjittedFnWithContext = partitioner.partition(self.model.to_bf16, (params_spec,), params_spec)

        # This will auto-magically run in mesh context
        self.params = p_shard_params(freeze(jax_utils.unreplicate(self.params)))  # type: ignore[no-untyped-call]
        self.is_sharded = True

        def generate(
            params: dict[Any, Any],
            input_features: ND_ARRAY_FLOAT,
            forced_decoder_ids: DECODER_ID_TYPE,
            return_timestamps: bool,
        ) -> FlaxBeamSearchOutput | FlaxGreedySearchOutput:
            output_ids: FlaxBeamSearchOutput | FlaxGreedySearchOutput = self.model.pipeline_generate(
                input_features,
                params=params,
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps=return_timestamps,
                max_length=self.max_length,
            )  # type: ignore[no-untyped-call]
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
        input_features: ND_ARRAY_FLOAT,
        language: str | None = None,
        task: JAXValidTasks | None = None,
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
                freeze(self.params), input_features, forced_decoder_ids, return_timestamps
            ).sequences
        return output_ids

    def get_forced_decoder_ids(
        self,
        generation_config: GenerationConfig | None = None,
        task: JAXValidTasks | None = None,
        language: str | LanguageIDs | None = None,
        return_timestamps: bool = False,
    ) -> DECODER_ID_TYPE:
        _generation_config: GenerationConfig = (
            self.model.generation_config if generation_config is None else generation_config
        )

        if hasattr(_generation_config, "is_multilingual"):
            is_multilingual = _generation_config.is_multilingual
        else:
            is_multilingual = None

        forced_decoder_ids: DECODER_ID_TYPE = []

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
                        acceptable_languages: list[str] = list(TO_LANGUAGE_CODE.values())
                    elif "<" in language or "|" in language or ">" in language:
                        # generation config language code
                        acceptable_languages = list(_generation_config.lang_to_id.keys())
                    else:
                        # language passed as a string
                        acceptable_languages = list(TO_LANGUAGE_CODE.keys())
                    msg = f"Unsupported language: {language}. Language should be one of: {acceptable_languages}."
                    raise ValueError(msg)
                forced_decoder_ids.append((1, _generation_config.lang_to_id[language_token]))

            if task is not None:
                forced_decoder_ids.append((2, _generation_config.task_to_id[task]))
            else:
                forced_decoder_ids.append((2, _generation_config.task_to_id["transcribe"]))

        if not return_timestamps:
            if forced_decoder_ids and forced_decoder_ids[-1][0] != _generation_config.no_timestamps_token_id:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, _generation_config.no_timestamps_token_id))

        return forced_decoder_ids

    def chunk_iter_with_batch(
        self, inputs: VEC_FLOAT, chunk_len: int, stride_left: int, stride_right: int, batch_size: int
    ) -> Generator[dict[str, list[tuple[int, int, int] | Any]], Any, None]:
        inputs_len: int = inputs.shape[0]  # Size of 1-D input / length of 1-channel audio array
        step: int = chunk_len - stride_left - stride_right

        all_chunks_start_idx: VEC_SIGNED_INT = np.arange(0, inputs_len, step)
        num_samples: int = len(all_chunks_start_idx)

        num_batches: int = math.ceil(num_samples / batch_size)
        batch_idx: list[VEC_SIGNED_INT] = np.array_split(np.arange(num_samples), num_batches)

        for _i, idx in enumerate(batch_idx):
            chunk_start_idx: VEC_SIGNED_INT = all_chunks_start_idx[idx]

            chunk_end_idx: VEC_SIGNED_INT = chunk_start_idx + chunk_len

            chunks: list[VEC_FLOAT] = [
                inputs[chunk_start:chunk_end] for chunk_start, chunk_end in zip(chunk_start_idx, chunk_end_idx)
            ]
            processed: BatchFeature = self.feature_extractor(
                chunks, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="np"
            )

            _stride_left: VEC_SIGNED_INT = np.where(chunk_start_idx == 0, 0, stride_left)
            is_last: VEC_SIGNED_INT = np.where(
                stride_right > 0, chunk_end_idx > inputs_len, chunk_end_idx >= inputs_len
            )
            _stride_right: VEC_SIGNED_INT = np.where(is_last, 0, stride_right)

            chunk_lens: list[int] = [chunk.shape[0] for chunk in chunks]
            strides: list[tuple[int, int, int]] = [
                (chunk_l, _stride_l, _stride_r)
                for chunk_l, _stride_l, _stride_r in zip(chunk_lens, _stride_left, _stride_right)
            ]

            yield {"stride": strides, **processed}

    def preprocess_batch(
        self,
        inputs: list[bytes] | bytes,
        chunk_length_s: float = 30.0,
        stride_length_s: float | None = None,
        batch_size: int | None = None,
        sampling_rate: int = 16000,
        backend: Parallel = JAXThreadParallel,
    ) -> Generator[BatchFeature | dict[str, list[tuple[int, int, int] | Any]], Any, None]:
        # runtime
        self.validate_data(inputs)

        _batch_size: int = batch_size if batch_size is not None else self.batch_size

        _inputs, _converted_sampling_rate = self.preprocess_input_ffmpeg(
            data=[inputs] if isinstance(inputs, bytes) else inputs, sampling_rate=sampling_rate, backend=backend
        )

        target_sampling_rate: int = self.feature_extractor.sampling_rate

        if _converted_sampling_rate != target_sampling_rate:
            try:
                import librosa
            except ImportError as err:
                msg: str = "To support resampling audio files, please install 'librosa' and 'soundfile'."
                raise ImportError(msg) from err
            _inputs, ratio = self.resample_input_with_fn(
                data=_inputs,
                resampler_fn=librosa.resample,
                in_sampling_rate=_converted_sampling_rate,
                out_sampling_rate=target_sampling_rate,
            )
        else:
            ratio = 1.0

        for idx, _input in enumerate(_inputs):
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

                yield from self.chunk_iter_with_batch(
                    _input,
                    chunk_len,
                    stride_left,
                    stride_right,
                    _batch_size,
                )
            else:
                # We don't need to chunk, so no striding as well
                processed: BatchFeature = self.feature_extractor(
                    _input, sampling_rate=target_sampling_rate, return_tensors="np"
                )
                yield processed

    def postprocess(
        self,
        model_outputs: list[dict[Any, list[Any]]],
        return_timestamps: bool | None = None,
        return_language: str | None = None,
    ) -> dict[str, Any]:
        # unpack the outputs from list(dict(list)) to list(dict)
        _model_outputs: list[dict[Any, Any]] = [
            dict(zip(output, t)) for output in model_outputs for t in zip(*output.values())
        ]

        time_precision: float = self.feature_extractor.chunk_length / self.model.config.max_source_positions
        # Send the chunking back to seconds, it's easier to handle in whisper
        sampling_rate: int = self.feature_extractor.sampling_rate
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
        return {"text": text, **optional}

    def forward(
        self,
        model_inputs: dict[Any, Any],
        batch_size: int | None = None,
        language: str | None = None,
        task: JAXValidTasks | None = None,
        return_timestamps: bool = False,
    ) -> dict[str, Any]:
        # We need to keep track of some additional input arguments for post-processing so need to forward these on after running generation
        input_features = model_inputs.pop("input_features")
        input_batch_size = input_features.shape[0]

        if input_batch_size != batch_size:
            padding = np.zeros([batch_size - input_batch_size, *input_features.shape[1:]], input_features.dtype)
            input_features = np.concatenate([input_features, padding])

        pred_ids = self.generate(input_features, language=language, task=task, return_timestamps=return_timestamps)[
            :input_batch_size
        ]

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
        task: JAXValidTasks = "transcribe",
        return_timestamps: bool = False,
    ) -> dict[str, Any]:
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

        Return:
            `Dict`: A dictionary with the following keys:
                - **text** (`str` ) -- The recognised text.
                - **chunks** (*optional(, `List[Dict]`)
                    When using `return_timestamps`, the `chunks` will become a list containing all the various text
                    chunks identified by the model, *e.g.* `[{"text": "hi ", "timestamps": (0.5,0.9), {"text":
                    "there", "timestamps": (1.0, 1.5)}]`. The original full text can roughly be recovered by doing
                    `"".join(chunk["text"] for chunk in output["chunks"])`.
        """
        self._check_is_initialized()

        batch_size = batch_size if batch_size is not None else self.batch_size
        if batch_size % self.min_batch_size != 0:
            msg = f"Batch size must be a multiple of the number of JAX devices, but got batch size {batch_size} and num devices {self.min_batch_size}."
            raise ValueError(msg)

        dataloader = self.preprocess_batch(
            inputs, chunk_length_s=chunk_length_s, stride_length_s=stride_length_s, batch_size=batch_size
        )
        model_outputs = []
        # iterate over our chunked audio samples
        for batch in dataloader:
            model_outputs.append(
                self.forward(
                    batch, batch_size=batch_size, language=language, task=task, return_timestamps=return_timestamps
                )
            )
        post_processed = self.postprocess(model_outputs, return_timestamps=return_timestamps)
        return post_processed

    def initialize_pipeline(
        self,
        language: str = "english",
        task: JAXValidTasks = "transcribe",
        return_timestamps: bool = False,
        shard_with: SHARDING_RULESET_TYPE | Literal[True] | None = None,
        use_experimental_cache: bool = False,
    ) -> None:
        """instantiate and return the Pipeline with internal batching, meant for large files that can be chunked internally.
        sets `~JAXPipeline.model` after initialising it.

        Args:
            language (str):
                The language to perform the task in, defaults to `english`
            task (ValidJaxTasks):
                The mode of the task, one of `transcribe` or `translate`, defaults to `transcribe`
            return_timestamps (bool):
                whether to return the timestamps of the generated text.
            use_experimental_cache (bool):
                use `experimental.compilation_cache` for compilation phase, defaults to False.

        Raises:
            ValueError: if neither of `sample_data: bytes` cannot be read.

        Returns:
            None
        """
        self.logger.info(f"Initializing {self.checkpoint}/{self.dtype} pipeline")

        if use_experimental_cache:
            cc.initialize_cache(JAX_CACHE_PATH)  # type: ignore[no-untyped-call]

        _random_inputs = {"input_features": np.ones((self.batch_size, 80, 3000))}
        self.logger.info(f"Compiling {self.checkpoint}/{self.dtype} pipeline")
        start: float = time.time()

        _shard_with = FLAX_DEFAULT_DP_LOGICAL_AXES if shard_with == True else shard_with

        if _shard_with is not None and self.device_type != "CPU":
            self.shard_params(num_mp_partitions=self.num_devices, logical_axis_rules=_shard_with)

        _: dict[str, Any] = self.forward(
            _random_inputs,
            batch_size=self.batch_size,
            language=language,
            task=task,
            return_timestamps=return_timestamps,
        )
        self.initialized = True
        self.logger.info(f"Compilation done in {time.time() - start:.2f}s")

    @staticmethod  # type: ignore[misc]
    def preprocess_input_ffmpeg(
        data: list[bytes],
        sampling_rate: int = 16000,
        backend: Parallel = JAXThreadParallel,  # Always takes a list of inputs
    ) -> tuple[AUDIO_FILES_DTYPE, int]:
        # returns batch_size x VEC[data_length[float32]],for a single input batch_size will be 1
        # so returned array will have a shape of (batches, data_length)
        converted: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.array(
            backend(delayed(ffmpeg_read)(d, sampling_rate=sampling_rate) for d in data)
        )
        return converted, sampling_rate

    @staticmethod  # type: ignore[misc]
    def resample_input_with_fn(
        data: AUDIO_FILES_DTYPE,  #
        resampler_fn: Callable[..., AUDIO_FILES_DTYPE],
        in_sampling_rate: int,
        out_sampling_rate: int = 16000,
        backend: Parallel = JAXThreadParallel,
    ) -> tuple[AUDIO_FILES_DTYPE, float]:
        # accepts batch_size x VEC[data_length[float32]]
        # returns batch_size x VEC[data_length[float32]]
        converted: AUDIO_FILES_DTYPE = np.array(
            backend(delayed(resampler_fn)(d, orig_sr=in_sampling_rate, target_sr=out_sampling_rate) for d in data)
        )
        return converted, float(in_sampling_rate) / float(out_sampling_rate)
