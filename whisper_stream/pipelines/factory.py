from typing import Any, Callable, Literal
from whisper_stream.pipelines.pipelines_jax import initialize_impl_batched_jax_pipeline, initialize_batched_jax_pipeline

BackendOpts = Literal["jax"]
JaxMethodOpts = Literal["implicit_batching", "explicit_batching"]

jax_methods: dict[JaxMethodOpts, Callable[..., Any]] = {
    "implicit_batching": initialize_impl_batched_jax_pipeline,
    "explicit_batching": initialize_batched_jax_pipeline,
}

methods: dict[BackendOpts, dict[JaxMethodOpts, Callable[..., Any]]] = {"jax": jax_methods}


def pipeline_factory(backend: BackendOpts, method: JaxMethodOpts) -> Callable[..., Any]:
    return methods[backend][method]
