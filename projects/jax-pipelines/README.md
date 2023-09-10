# `whisper` pipelines in `Flax/JAX`

## Usage

### More on [`WHISPER-JAX`](../../whisper_stream/vendored/whisper_jax/README.md)

### [`Notebook`](../..//development/notebooks/usage-jax-whisper-streaming.ipynb)

### `Quickstart`

```python
from whisper_stream.core.helpers.data_loading import load_data_samples_from_path
from whisper_stream.projects.jax_pipelines import (
    JAXStreamingPipeline,
)
from whisper_stream.projects.jax_pipelines.constants import (
    JAXValidDtypesMapping,
    JAXScalarDType,
)

from whisper_stream.core.constants import (
    WhisperValidCheckpoints,
    WhisperValidTasks
)

from whisper_stream.core.logger import LogLevelNames
from pathlib import Path
import time

# Prepare
checkpoint: WhisperValidCheckpoints = "openai/whisper-tiny"
model_dtype: JAXScalarDType = JAXValidDtypesMapping["BFLOAT16"]
task: WhisperValidTasks = "transcribe"
language: str = "english"
return_timestamps: bool = True
batch_size: int = 1
log_level: LogLevelNames = "INFO"

data_directory = Path("../data")

run_opts = {"batch_size": batch_size, "return_timestamps": return_timestamps, "language": language, "task": task}

# construct
pipeline = JAXStreamingPipeline(
    checkpoint=checkpoint, dtype=model_dtype, batch_size=batch_size, min_log_level=log_level
)

# Load data
pipeline_data: bytes = load_data_samples_from_path("audio_2.mp3", directory=data_directory, binary_mode=True)  # 3s
pipeline_data_large: bytes = load_data_samples_from_path("tryst.mp3", directory=data_directory, binary_mode=True)  # 4:44s

# initialize & warmup
pipeline.initialize_pipeline(**run_opts, use_experimental_cache=True)

# small data
list(pipeline(pipeline_data, **run_opts))

# small data in batch
list(pipeline([pipeline_data] * 10, **run_opts))

# larger, chunkable data
list(pipeline(pipeline_data_large, **run_opts))

# alrger, chunkable data in batches
list(pipeline([pipeline_data_large] * 32, **run_opts))

# make mixed mode data
mixed_mode_data: list[bytes] = [pipeline_data_large, pipeline_data, pipeline_data, pipeline_data] * 4

# test on mixed data, data is received as it comes
# using default `smallest` strategy the smaller files will come in larger batches first
start: float = time.time()
for data in pipeline(mixed_mode_data, strategy="smallest", **run_opts):
    print({"num_items": len(data)}, end="\n")
    print({"data": data, "time_taken": f"{time.time() - start:.2}s"}, end="\n")
    print("-" * 40, end="\n")
    start = time.time()
```
