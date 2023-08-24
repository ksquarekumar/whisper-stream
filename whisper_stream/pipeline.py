import time
from pathlib import Path
from typing import Final, Literal

import jax.numpy as jnp
from jax._src.numpy.lax_numpy import _ScalarMeta as ScalarMeta

from whisper_stream.whisper_jax import FlaxWhisperPipeline

__FILE__: Final[str] = __file__

ValidDtypes: Final[dict[str, ScalarMeta]] = {"FLOAT32": jnp.float32, "BFLOAT16": jnp.bfloat16, "FLOAT16": jnp.float16}

ValidCheckpoints = Literal[
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
    "openai/whisper-large-v2",
]


def load_data_sample_from_path(sample_file: str) -> str:
    files = list(Path(__FILE__).parent.glob(f"data/{sample_file}"))
    if len(files) >= 1:
        return str(files[0].absolute())
    msg = f"File {sample_file} does not exist in ./data"
    raise FileNotFoundError(msg)


# 2D parameter and activation partitioning for DP
logical_axis_rules_dp = (
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


def initialize_jax_pipeline(
    checkpoint: ValidCheckpoints,
    sample_data_path: str,
    dtype: ScalarMeta = ValidDtypes["FLOAT32"],
    batch_size: int | None = None,
    max_length: int | None = None,
) -> FlaxWhisperPipeline:
    """instantiate and return the Pipeline.

    Args:
        checkpoint (`str`, *optional*, defaults to `"openai/whisper-large-v2"):
                The Whisper checkpoint to use with the pipeline. Must be an available checkpoint on the Hugging Face Hub
                with Flax weights.
        sample_data_path: Path to an audio file with which the model will be pre-compiled
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPU(s)). This can be used to enable half-precision inference on GPUs or TPUs.
            If specified all the computation will be performed with the given `dtype`. **Note that this only
            specifies the dtype of the computation and does not influence the dtype of model parameters.**
        batch_size (`int`, *optional*, defaults to the minimum per-device batch size, i.e. `jax.local_device_count()`):
            The batch size to be used in chunking transcription. Beneficial for transcribing long audio files. Passing
            a batch size in the `__init__` method will be superseded by any batch size passed to the `__call__` method.
        max_length (`int`, *optional*):
            The maximum numbers of tokens to generate. Defaults to `model.config.max_length`.

    Returns:
        FlaxWhisperPipeline: an instance of `FlaxWhisperPipeline` pre-initialized with data
    """
    # instantiate pipeline
    pipeline: FlaxWhisperPipeline = FlaxWhisperPipeline(checkpoint=checkpoint, dtype=dtype, batch_size=batch_size, max_length=max_length)  # type: ignore[no-untyped-call]
    # optimize for data parallelism
    pipeline.shard_params(num_mp_partitions=1, logical_axis_rules=logical_axis_rules_dp)  # type: ignore[no-untyped-call]
    # compile
    pipeline(load_data_sample_from_path(sample_data_path))
    # return
    return pipeline


if __name__ == "__main__":
    model: Final[ValidCheckpoints] = "openai/whisper-tiny"
    pipeline: FlaxWhisperPipeline = initialize_jax_pipeline(model, "audio_1.mp3")
    start: float = time.time()
    print("Output:", pipeline(load_data_sample_from_path("audio_1.mp3")))  # noqa: T201
    print(f"Time taken: {time.time() - start}")  # noqa: T201
