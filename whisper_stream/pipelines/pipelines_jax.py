#!python3
import time
from typing import Any, Callable, Final, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.numpy.lax_numpy import _ScalarMeta as ScalarMeta
from jax._src.basearray import Array
from whisper_stream.utils.helpers import is_bytes, is_bytes_array, parse_known_kwargs, language_ids

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

ValidJaxTasks = Literal["transcribe", "translate"]

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
    pipeline_kwargs: dict[str, Any] | None = None,
    language: str = "english",
    task: ValidJaxTasks = "transcribe",
    return_timestamps: bool = False,
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
        language (str):
            The language to perform the task in, defaults to `english`
        task (ValidJaxTasks):
            The mode of the task, one of `transcribe` or `translate`, defaults to `transcribe`
        return_timestamps (bool):
            whether to return the timestamps of the generated text.

    Kwargs:
        pipeline_kwargs (Any):
            kwargs accepted by `FlaxWhisperPipeline`, passes only known `kwargs`

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
    known_kwargs: dict[str, Any] = parse_known_kwargs(func_or_class=FlaxWhisperPipeline, kwargs=pipeline_kwargs or {})

    # log
    binds: dict[str, str] = {"model": str(checkpoint), "mode": MODE}
    logger.info(
        "Initializing pipeline",
        **binds,
        sample_data=f"bytes:(size={len(sample_data)})",
        dtype=dtype,
        language=language,
        task=task,
        return_timestamps=return_timestamps,
        **known_kwargs,
    )
    # instantiate pipeline class
    pipeline: FlaxWhisperPipeline = FlaxWhisperPipeline(checkpoint=checkpoint, dtype=dtype, **known_kwargs)  # type: ignore[no-untyped-call]

    # optimize for data parallelism
    pipeline.shard_params(num_mp_partitions=1, logical_axis_rules=logical_jax_axis_rules_for_dp)  # type: ignore[no-untyped-call]
    # compile
    start: float = time.time()
    logger.info("starting pre-compilation", **binds)
    pipeline(sample_data, language=language, task=task, return_timestamps=return_timestamps)
    logger.info("finished pre-compilation", **binds, time_taken=f"{(time.time() - start):.2f}s")
    # return
    return pipeline


def initialize_batched_jax_pipeline(
    checkpoint: ValidJaxCheckpoints,
    sample_data: list[bytes],
    dtype: ScalarMeta | jnp.dtype[Any] = ValidDtypes["FLOAT32"],
    task: Literal["transcribe", "translate"] = "transcribe",
    language: str = "<|en|>",
    return_timestamps: bool = False,
    is_multilingual: bool = False,
    max_new_tokens: int | None = None,
    min_new_tokens: int | None = None,
    pretrained_tokenizer_model_name: ValidJaxCheckpoints | None = None,
    generator_kwargs: dict[str, Any] | None = None,
    processor_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
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
        language (str, optional)
            language to use for translation, defaults to "<|en|>"
        return_timestamps (bool, optional)
            whether to return timestamps in transcriptions, defaults to False
        is_multilingual (bool):
            whether the language model is initialized as `is_multilingual`, defaults to True
        max_new_tokens (int, optional):
            number of tokens, defaults to None
            if not specified, defaults to model.config.maxlength
        min_new_tokens (int, optional):
            number of tokens, defaults to None
        pretrained_tokenizer_model_name (str, optional):
            name of the tokenizer to use, defaults to `checkpoint`
    Kwargs:
        generator_kwargs (Any):
            kwargs accepted by `FlaxWhisperForConditionalGeneration.from_pretrained`, passes only known `kwargs`
        processor_kwargs (Any):
            kwargs accepted by `WhisperProcessor.from_pretrained`, passes only known `kwargs`
        model_kwargs (Any):
            kwargs accepted by `FlaxWhisperForConditionalGeneration.from_pretrained().generate`, passes only known `kwargs`

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

    # check language-id
    if language not in language_ids:
        message = f"language {language} not supported, must be one of {language_ids}"
        raise ValueError(message)

    # check and parse kwargs
    generator_known_kwargs: dict[str, Any] = parse_known_kwargs(
        func_or_class=FlaxWhisperForConditionalGeneration.from_pretrained, kwargs=generator_kwargs or {}
    )
    processor_known_kwargs: dict[str, Any] = parse_known_kwargs(
        func_or_class=WhisperProcessor.from_pretrained, kwargs=processor_kwargs or {}
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
    model_generation_known_kwargs: dict[str, Any] = parse_known_kwargs(
        func_or_class=model.generate, kwargs=model_kwargs or {}
    )

    def get_model_generation_configuration(
        task: str = task,
        language: str = language,
        return_timestamps: bool = return_timestamps,
        max_new_tokens: int | None = max_new_tokens,
        min_new_tokens: int | None = min_new_tokens,
        model_generation_known_kwargs: dict[str, Any] = model_generation_known_kwargs,
    ) -> dict[str, Any]:
        """
        Returns:
            dict[str, Any]: model generation configuration
        """

        model_generation_config: dict[str, Any] = {
            **model_generation_known_kwargs,
            "task": task,
            "language": language,
            "is_multilingual": is_multilingual,
            "return_timestamps": return_timestamps,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
        }

        if model_generation_config["max_new_tokens"] is None:
            model_generation_config["max_length"] = model.config.max_length

        return {k: v for k, v in model_generation_config.items() if v is not None}

    config: dict[str, Any] = get_model_generation_configuration()

    def generate_fn(input_features: Array) -> Array:
        # Generate tokenized outputs
        pred_ids: FlaxGreedySearchOutput | FlaxSampleOutput | FlaxBeamSearchOutput = model.generate(
            input_features,
            params=params,
            task=config.pop("task", None),
            return_timestamps=config.pop("return_timestamps", None),
            language=config.pop("language", None),
            is_multilingual=config.pop("is_multilingual", True),
            max_length=config.pop("max_length", None),
            max_new_tokens=config.pop("max_new_tokens", None),
            min_new_tokens=config.pop("min_new_tokens", None),
        )

        return pred_ids.sequences  # type: ignore[no-any-return]

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
        **model_generation_known_kwargs,
    )

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

        # devices x batch_size x feature_dims for GPU / batch_size x 1 x feature_dims for CPU
        input_features: np.ndarray[Any, Any] = (
            shard(batch["input_features"]) if mode == "GPU" else batch["input_features"]  # type: ignore[no-untyped-call]
        )

        return input_features

    def postprocess(pred_ids: np.ndarray[Any, Any], skip_special_tokens: bool = True) -> list[str]:
        transcriptions: list[str] = processor.batch_decode(pred_ids, skip_special_tokens=skip_special_tokens)

        return transcriptions

    # warm up
    start: float = time.time()
    logger.info("starting pre-compilation", **binds)

    # get all input features
    input_features: np.ndarray[Any, Any] = preprocess(sample_data)

    # do forward pass
    if mode == "GPU":
        # replicate the model params across devices, with cpu we should already have it
        jax_utils.replicate(params)  # type: ignore[no-untyped-call]

    warm_up_pred_ids: np.ndarray[Any, Any] = parallel_mapped_generate_fn(input_features)

    # post-process: convert tokens ids to text string
    transcription: list[str] = postprocess(warm_up_pred_ids)

    if len(transcription) == 0:
        message = "Could not initialize model"
        raise RuntimeError(message)
    logger.info(
        "finished pre-compilation", **binds, transcription=transcription, time_taken=f"{(time.time() - start):.2f}s"
    )

    def pipeline(
        data: list[bytes],
        *,
        parallelize: bool = True,
    ) -> list[str]:
        """
        Initialized pipeline and preprocessor

        Args:
            data: (list[bytes])
                list of data in `bytes`
            parallelize: (bool)
                whether to parallelize the pipeline or not, defaults to `False`

        Returns:
            str, prediction
        """
        # get all input features
        input_features: np.ndarray[Any, Any] = preprocess(data)

        if parallelize:
            # generate tokenized outputs
            parallel_mapped_pred_ids: list[np.ndarray[Any, Any]] = ThreadParallel(
                delayed(parallel_mapped_generate_fn)(input_feature.reshape(1, *input_feature.shape))
                for input_feature in input_features
            )
            # post-process: convert tokens ids to text
            transcriptions: list[str] = ThreadParallel(
                delayed(postprocess)(pred_id, skip_special_tokens=True) for pred_id in parallel_mapped_pred_ids
            )
        else:
            # generate tokenized outputs
            pred_ids: np.ndarray[Any, Any] = parallel_mapped_generate_fn(
                input_features,
            )
            # post-process: convert tokens ids to text
            transcriptions = postprocess(pred_ids, skip_special_tokens=True)

        return transcriptions

    return pipeline


__all__: list[str] = ["ValidJaxCheckpoints", "initialize_impl_batched_jax_pipeline", "initialize_batched_jax_pipeline"]

if __name__ == "__main__":
    model: Final[ValidJaxCheckpoints] = "openai/whisper-tiny"
    sample_data: bytes = load_data_sample_from_path("audio_1.mp3", binary_mode=True)
    # # Test for all
    # # (1) ImplicitBatchedJAX
    # logger.info("Testing: (1)`initialize_impl_batched_jax_pipeline`")
    # impl_pipeline: FlaxWhisperPipeline = initialize_impl_batched_jax_pipeline(checkpoint=model, sample_data=sample_data)
    # start: float = time.time()
    # logger.info("Output", output=impl_pipeline(sample_data), time_taken=f"{(time.time() - start):.2f}s")
    # (2) ExplicitBatchedJAX
    logger.info("Testing: (2)`initialize_explicit_batched_jax_pipeline`")
    explicit_pipeline: Callable[[list[bytes]], list[str]] = initialize_batched_jax_pipeline(
        checkpoint=model, sample_data=[sample_data] * 5
    )
    start = time.time()
    logger.info("Output", output=explicit_pipeline([sample_data] * 10), time_taken=f"{(time.time() - start):.2f}s")
