#!python3
import time
from typing import Any, Callable, Final, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.numpy.lax_numpy import _ScalarMeta as ScalarMeta
from jax._src.basearray import Array
from whisper_stream.utils.helpers import is_bytes, is_bytes_array, parse_known_kwargs

from flax import jax_utils

from whisper_stream.whisper_jax import FlaxWhisperPipeline
from whisper_stream.data.prefetch import load_data_sample_from_path
from whisper_stream.logger import get_application_logger, BoundLogger
from transformers import FlaxWhisperForConditionalGeneration, WhisperProcessor
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from transformers.feature_extraction_utils import BatchFeature
from flax.training.common_utils import shard

from joblib import Parallel, delayed, cpu_count

__FILE__: Final[str] = __file__

ThreadParallel = Parallel(backend="threading", n_jobs=cpu_count())

logger: BoundLogger = get_application_logger(name="pipeline")
ValidDtypes: Final[dict[str, ScalarMeta]] = {"FLOAT32": jnp.float32, "BFLOAT16": jnp.bfloat16, "FLOAT16": jnp.float16}

ValidJaxCheckpoints = Literal[
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
    "openai/whisper-large-v2",
]


# 2D parameter and activation partitioning for DP
logical_jax_axis_rules_for_dp = (
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


def initialize_impl_batched_jax_pipeline(
    checkpoint: ValidJaxCheckpoints,
    sample_data: bytes,
    dtype: ScalarMeta = ValidDtypes["FLOAT32"],
    **kwargs: Any,
) -> FlaxWhisperPipeline:
    """instantiate and return the Pipeline with internal batching, meant for large files that can be chunked.

    Args:
        checkpoint (str):
            The Whisper checkpoint to use with the pipeline. Must be an available checkpoint on the Hugging Face Hub
            with Flax weights.
        sample_data (bytes):
            read from the audio file, must be a valid `mp3` with correct sample rate.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPU(s)). This can be used to enable half-precision inference on GPUs or TPUs.
            If specified all the computation will be performed with the given `dtype`.
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    Kwargs:
        Any
            Kwargs accepted by `FlaxWhisperPipeline`

    Raises:
        ValueError: if neither of `sample_data: bytes` or `sample_data: str(Path)` are provided

    Returns:
        FlaxWhisperPipeline: an instance of `FlaxWhisperPipeline` pre-initialized with data
    """
    MODE: Final[str] = "whisper-jax-implicit-batching"
    # check for sample data
    if not is_bytes(sample_data):
        message: str = f"data must be of type bytes, received {type(sample_data)}"
        raise ValueError(message)
    # check and parse kwargs
    known_kwargs: dict[str, Any] = parse_known_kwargs(func_or_class=FlaxWhisperPipeline, kwargs=kwargs)

    # log
    binds: dict[str, str] = {"model": str(checkpoint), "mode": MODE}
    logger.info(
        "Initializing pipeline", **binds, sample_data=f"bytes:(size={len(sample_data)})", dtype=dtype, **known_kwargs
    )
    # instantiate pipeline class
    pipeline: FlaxWhisperPipeline = FlaxWhisperPipeline(checkpoint=checkpoint, dtype=dtype, **known_kwargs)  # type: ignore[no-untyped-call]
    # optimize for data parallelism
    pipeline.shard_params(num_mp_partitions=1, logical_axis_rules=logical_jax_axis_rules_for_dp)  # type: ignore[no-untyped-call]
    # compile
    start: float = time.time()
    logger.info("starting pre-compilation", **binds)
    pipeline(sample_data)
    logger.info("finished pre-compilation", **binds, time_taken=f"{(time.time() - start):.2f}s")
    # return
    return pipeline


def initialize_batched_jax_pipeline(
    checkpoint: ValidJaxCheckpoints,
    sample_data: list[bytes],
    dtype: ScalarMeta | jnp.dtype[Any] = ValidDtypes["FLOAT32"],
    task: Literal["transcribe", "translate"] = "transcribe",
    return_timestamps: bool = False,
    max_new_tokens: int = 25,
    min_new_tokens: int = 25,
    pretrained_tokenizer_model_name: ValidJaxCheckpoints | None = None,
    **kwargs: Any,
) -> Callable[[list[bytes]], list[str]]:
    """instantiate and return the Pipeline with explicit batching, meant for multiple smaller files [<30s].

    Args:
        checkpoint (str):
            The Whisper checkpoint to use with the pipeline. Must be an available checkpoint on the Hugging Face Hub
            with Flax weights.
        sample_data (list[bytes]):
            read from array of audio file(s), must be a valid `mp3` with correct sample rate.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPU(s)). This can be used to enable half-precision inference on GPUs or TPUs.
            If specified all the computation will be performed with the given `dtype`.
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
        task: (Literal["transcribe", "translate"], optional)=
            whether to perform transcription or translation, defaults to "transcribe"
        return_timestamps (bool, optional)
            whether to return timestamps in transcriptions, defaults to False
        max_new_tokens (int, optional):
            number of tokens, defaults to 25
        min_new_tokens (int, optional):
            number of tokens, defaults to 25
        pretrained_tokenizer_model_name (str, optional):
            name of the tokenizer to use, defaults to `checkpoint`
    Kwargs:
        Any
            Kwargs accepted by `FlaxWhisperForConditionalGeneration`

    Raises:
        ValueError: if neither of `sample_data: bytes` or `sample_data: str(Path)` are provided

    Returns:
        FlaxWhisperForConditionalGeneration: an instance of `FlaxWhisperForConditionalGeneration` pre-initialized with data
    """
    MODE: Final[str] = "whisper-jax-explicit-batching"
    # check for sample data
    if not is_bytes_array(sample_data):
        message: str = f"data must be an list of bytes, received {type(sample_data)}"
        raise ValueError(message)

    # check and parse kwargs
    generator_known_kwargs: dict[str, Any] = parse_known_kwargs(
        func_or_class=FlaxWhisperForConditionalGeneration.from_pretrained, kwargs=kwargs
    )
    processor_known_kwargs: dict[str, Any] = parse_known_kwargs(
        func_or_class=WhisperProcessor.from_pretrained, kwargs=kwargs
    )

    # load model and params
    model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=checkpoint, dtype=dtype, _do_init=False, **generator_known_kwargs
    )

    # load tokenizer
    processor: WhisperProcessor = WhisperProcessor.from_pretrained(
        pretrained_tokenizer_model_name or checkpoint, **processor_known_kwargs
    )

    # parse kwargs for model.generate
    model_generation_known_kwargs: dict[str, Any] = parse_known_kwargs(func_or_class=model.generate, kwargs=kwargs)

    # log
    binds: dict[str, str] = {"model": str(checkpoint), "mode": MODE}
    logger.info(
        "Initializing pipeline",
        **binds,
        sample_data=f"bytes array:(size={len(sample_data)})",
        dtype=dtype,
        task=task,
        return_timestamps=return_timestamps,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        **generator_known_kwargs,
        **processor_known_kwargs,
    )

    def generate_fn(input_features: Array) -> Array:
        # Generate tokenized outputs
        pred_ids: FlaxGreedySearchOutput | FlaxSampleOutput | FlaxBeamSearchOutput = model.generate(
            input_features,
            params=params,
            task=task,
            return_timestamps=return_timestamps,
            max_length=model.config.max_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            **model_generation_known_kwargs,
        )

        return pred_ids.sequences  # type: ignore[no-any-return]

    # set device flag
    if jax.devices()[0].device_kind != "cpu":
        mode: Literal["CPU", "GPU"] = "GPU"
    else:
        mode = "CPU"

    # If GPU / Multiple GPU's replicate, use `pmap` else use `jit`
    parallel_mapped_generate_fn = jax.pmap(generate_fn, "batch") if mode == "GPU" else jax.jit(generate_fn)

    def preprocess(data: list[bytes]) -> np.ndarray[Any, Any]:
        # batch_size x VEC[data_length]
        pre: np.ndarray[Any, Any] = np.array(ThreadParallel(delayed(ffmpeg_read)(d, sampling_rate=16000) for d in data))

        # batch_size x feature_dims
        batch: BatchFeature = processor(pre, sampling_rate=16000, return_tensors="np")

        # num_pools x batch_size x feature_dims
        input_features: np.ndarray[Any, Any] = shard(batch["input_features"]) if mode == "GPU" else batch["input_features"]  # type: ignore[no-untyped-call]

        return input_features

    # warm up
    start: float = time.time()
    logger.info("starting pre-compilation", **binds)

    # get all input features
    input_features: np.ndarray[Any, Any] = preprocess(sample_data)

    # do forward pass
    if mode == "GPU":
        # replicate the model params across devices
        jax_utils.replicate(params)  # type: ignore[no-untyped-call]

    warm_up_pred_ids: np.ndarray[Any, Any] = parallel_mapped_generate_fn(input_features)

    # post-process: convert tokens ids to text string
    transcription: list[str] = processor.batch_decode(warm_up_pred_ids, skip_special_tokens=True)

    if len(transcription) == 0:
        message = "Could not initialize model"
        raise RuntimeError(message)
    logger.info(
        "finished pre-compilation", **binds, transcription=transcription, time_taken=f"{(time.time() - start):.2f}s"
    )

    def pipeline(data: list[bytes]) -> list[str]:
        # get all input features
        input_features: np.ndarray[Any, Any] = preprocess(data)

        # generate tokenized outputs
        pred_ids: np.ndarray[Any, Any] = parallel_mapped_generate_fn(input_features)

        # post-process: convert tokens ids to text string
        transcriptions: list[str] = processor.batch_decode(pred_ids, skip_special_tokens=True)

        return transcriptions

    return pipeline


__all__: list[str] = ["ValidJaxCheckpoints", "initialize_impl_batched_jax_pipeline", "initialize_batched_jax_pipeline"]

if __name__ == "__main__":
    model: Final[ValidJaxCheckpoints] = "openai/whisper-tiny"
    sample_data: bytes = load_data_sample_from_path("audio_1.mp3", binary_mode=True)
    # Test for all
    # (1) ImplicitBatchedJAX
    logger.info("Testing: (1)`initialize_impl_batched_jax_pipeline`")
    impl_pipeline: FlaxWhisperPipeline = initialize_impl_batched_jax_pipeline(checkpoint=model, sample_data=sample_data)
    start: float = time.time()
    logger.info("Output", output=impl_pipeline(sample_data), time_taken=f"{(time.time() - start):.2f}s")
    # (2) ExplicitBatchedJAX
    logger.info("Testing: (2)`initialize_explicit_batched_jax_pipeline`")
    explicit_pipeline: Callable[[list[bytes]], list[str]] = initialize_batched_jax_pipeline(
        checkpoint=model, sample_data=[sample_data] * 10
    )
    start = time.time()
    logger.info("Output", output=explicit_pipeline([sample_data] * 10), time_taken=f"{(time.time() - start):.2f}s")
