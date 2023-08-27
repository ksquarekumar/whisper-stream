from typing import Any, Callable
from whisper_stream.pipelines.jax import initialize_batched_jax_pipeline

methods = {"batched_jax": initialize_batched_jax_pipeline}


def pipeline_factory(method: str) -> Callable[..., Any]:
    return methods[method]
