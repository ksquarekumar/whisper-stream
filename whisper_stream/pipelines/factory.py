from typing import Any, Callable
from whisper_stream.pipelines.pipelines_jax import initialize_impl_batched_jax_pipeline, initialize_batched_jax_pipeline

methods = {
    "implicit_batched_jax": initialize_impl_batched_jax_pipeline,
    "explicit_batched_jax": initialize_batched_jax_pipeline,
}


def pipeline_factory(method: str) -> Callable[..., Any]:
    return methods[method]  # type: ignore[return-value]
