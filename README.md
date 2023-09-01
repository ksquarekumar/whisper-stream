[![CodeQL](https://github.com/ksquarekumar/whisper-stream/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ksquarekumar/whisper-stream/actions/workflows/github-code-scanning/codeql) [![Release](https://github.com/ksquarekumar/whisper-stream/actions/workflows/Release.yml/badge.svg)](https://github.com/ksquarekumar/whisper-stream/actions/workflows/Release.yml) ![Builds](https://codebuild.ap-south-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiS2FMcnRKSWhrNE0zYk0wR3dBRzlQSWVjQVBsbHhsYmwySWt6SG9zU1NRVWdrN1ZkTjJLNi83R1JPd3NWaDM5eU9sS0hVUUd4ODdUSGZ2Z3NCajZQbGNBPSIsIml2UGFyYW1ldGVyU3BlYyI6InFIYTNab2s1a3oxdWJVTnYiLCJtYXRlcmlhbFNldFNlcmlhbCI6Mn0%3D&branch=main)

# Whisper Stream

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

### From Git
`cuda_variant` can be either of `cuda11_local` or `cuda11_pip`
```console
pip install --upgrade "jax[{cuda_variant}]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "whisper-stream[{dependency}] @ git+https://github.com/ksquarekumar/whisper-stream.git@main"
```

### Installing with dependendcies

- install `pyenv`
```console
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile
exec "$SHELL"
```

- installing `conda/mamba` with `pyenv`
```console
pyenv install mambaforge-22.9.0-3
mamba activate base
```

- creating `whisper_py311` environment with `conda/mamba`
```console
pyenv install mambaforge-22.9.0-3
mamba activate base
mamba env create -f conda.yml
pip install .
mamba activate whisper_py311
```


### Installing in Development Mode with dependendcies

- install `pyenv`
```console
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile
exec "$SHELL"
```

- installing `conda/mamba` with `pyenv`
```console
pyenv install mambaforge-22.9.0-3
mamba activate base
```

- creating `whisper_py311` environment with `conda/mamba`
```console
pyenv install mambaforge-22.9.0-3
mamba activate base
mamba env create -f conda.yml
pip install -e ."[data,benchmarks,dev,test]"
mamba activate whisper_py311
pre-commit install --install-hooks
```

## [Whisper-JAX](./WhisperJax.md)

> `jax` modules partially vendored from [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax)

## Usage

### `JAX` Pipelines [Link](./notebooks/usage.ipynb)

- `quickstart`

```python
from whisper_stream import load_data_samples_from_path
from whisper_stream.pipelines import (
    JAXStreamablePipeline,
    JAXValidDtypesMapping,
    JAXScalarDType,
    JAXCheckpoints,
    JAXValidTasks,
)
from whisper_stream.logger import LOG_LEVEL_NAMES

# set parameters
checkpoint: JAXCheckpoints = "openai/whisper-tiny"
model_dtype: JAXScalarDType = JAXValidDtypesMapping["BFLOAT16"]
task: JAXValidTasks = "transcribe"
language: str = "english"
return_timestamps: bool = True
batch_size: int = 32
log_level: LOG_LEVEL_NAMES = "INFO"

run_opts = {"batch_size": batch_size, "return_timestamps": return_timestamps, "language": language, "task": task}

# construct
pipeline = JAXStreamablePipeline(
    checkpoint=checkpoint, dtype=model_dtype, batch_size=batch_size, min_log_level=log_level
)

# Load data
pipeline_data: bytes = load_data_samples_from_path("audio_1.mp3", binary_mode=True)  #4s
pipeline_data_large: bytes = load_data_samples_from_path("tryst.mp3", binary_mode=True) #4:44s

# initialize & warmup
pipeline.initialize_pipeline(**run_opts, use_experimental_cache=True)

# small data, one at a time
list(pipeline(pipeline_data, **run_opts))

# small data in batch
list(pipeline([pipeline_data] * 10, **run_opts))

# larger, chunkable data
%time list(pipeline(pipeline_data_large, **run_opts))

# alrger, chunkable data in batches
%time list(pipeline([pipeline_data_large] * 32, **run_opts))

# make mixed mode data
mixed_mode_data: list[bytes] = [pipeline_data_large, pipeline_data, pipeline_data, pipeline_data] * 4

# test on mixed data, data isreceived as it comes
# using default `smallest` strategy the smaller files will come in larger batches first
start: float = time.time()
for data in pipeline(mixed_mode_data, strategy="smallest", **run_opts):
    print({"num_items": len(data)}, end="\n")
    print({"data": data, "time_taken": f"{time.time() - start:.2}s"}, end="\n")
    print("-" * 40, end="\n")
    start = time.time()
```

## License

`whisper-stream` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
