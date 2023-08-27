#!python3
import time
from typing import Any, Final, Literal

import jax.numpy as jnp
from jax._src.numpy.lax_numpy import _ScalarMeta as ScalarMeta

from whisper_stream.whisper_jax import FlaxWhisperPipeline
from whisper_stream.data.prefetch import load_data_sample_from_path
from whisper_stream.logger import get_application_logger, BoundLogger

__FILE__: Final[str] = __file__

logger: BoundLogger = get_application_logger(name="pipeline")
ValidDtypes: Final[dict[str, ScalarMeta]] = {"FLOAT32": jnp.float32, "BFLOAT16": jnp.bfloat16, "FLOAT16": jnp.float16}

ValidCheckpoints = Literal[
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


def initialize_batched_jax_pipeline(
    checkpoint: ValidCheckpoints,
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
        sample_data_path (str, optional):
            path to an audio file with which the model will be pre-compiled, Optional
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
    # check for sample data
    data_load: str | bytes | None = None
    if not sample_data or not isinstance(sample_data, bytes):
        message: str = "Cannot initialize without data"
        raise ValueError(message)

    # instantiate pipeline
    binds: dict[str, str] = {"model": str(checkpoint), "mode": "whisper-jax"}
    logger.info(
        "Initializing pipeline",
        **binds,
        sample_data=f"bytes:(size={len(sample_data)})",
        dtype=dtype,
        **kwargs,
    )
    pipeline: FlaxWhisperPipeline = FlaxWhisperPipeline(checkpoint=checkpoint, dtype=dtype, **kwargs)  # type: ignore[no-untyped-call]
    # optimize for data parallelism
    pipeline.shard_params(num_mp_partitions=1, logical_axis_rules=logical_jax_axis_rules_for_dp)  # type: ignore[no-untyped-call]
    # compile
    start: float = time.time()
    logger.info("starting pre-compilation", **binds)
    pipeline(data_load)
    logger.info("finished pre-compilation", **binds, time_taken=f"{(time.time() - start):.2f}s")
    # return
    return pipeline


if __name__ == "__main__":
    model: Final[ValidCheckpoints] = "openai/whisper-tiny"
    sample_data: bytes = load_data_sample_from_path("audio_1.mp3", binary_mode=True)
    # Test for all
    # (1) BatchedJAX
    logger.info("Testing: (1)`initialize_batched_jax_pipeline`")
    pipeline: FlaxWhisperPipeline = initialize_batched_jax_pipeline(checkpoint=model, sample_data=sample_data)
    start: float = time.time()
    logger.info("Output", output=pipeline(sample_data), time_taken=f"{(time.time() - start):.2f}s")
