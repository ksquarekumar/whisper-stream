[![CodeQL](https://github.com/ksquarekumar/whisper-stream/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ksquarekumar/whisper-stream/actions/workflows/github-code-scanning/codeql) [![Release](https://github.com/ksquarekumar/whisper-stream/actions/workflows/Release.yml/badge.svg)](https://github.com/ksquarekumar/whisper-stream/actions/workflows/Release.yml) ![Builds](https://codebuild.ap-south-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiS2FMcnRKSWhrNE0zYk0wR3dBRzlQSWVjQVBsbHhsYmwySWt6SG9zU1NRVWdrN1ZkTjJLNi83R1JPd3NWaDM5eU9sS0hVUUd4ODdUSGZ2Z3NCajZQbGNBPSIsIml2UGFyYW1ldGVyU3BlYyI6InFIYTNab2s1a3oxdWJVTnYiLCJtYXRlcmlhbFNldFNlcmlhbCI6Mn0%3D&branch=main)

# Whisper Stream

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

### From Git

```console
pip install "whisper-stream[data] @ git+https://github.com/ksquarekumar/whisper-stream.git"
```

### In Development Mode

```console
conda install mamba
mamba env create -f conda.yml
mamba activate whisper_py310
virtualenv ./.venv
source ./.venv/bin/activate
pip install -e ."[data,dev,test]"
pre-commit install --install-hooks && pre-commit autoupdate
```

## [Whisper-JAX](./WhisperJax.md)

> `jax` modules partially vendored from [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax)

## Usage

### `JAX` Pipelines

### Implicit Batching _(for Large files)_

```python
from whisper_stream.pipelines import pipeline_factory, ValidJaxCheckpoints
from whisper_stream import load_data_sample_from_path

model: ValidJaxCheckpoints = "openai/whisper-tiny"
data: bytes = load_data_sample_from_path("audio_1.mp3", True)

# Warm-up
implicit_pipeline = pipeline_factory("jax","implicit_batching")(model, data)
```

```shell
$>
b'{"application":"whisper_stream","version":"0.0.1","python_version":"3.10.12","platform_architecture":["64bit",""],"model":"openai/whisper-tiny","mode":"whisper-jax-implicit-batching","sample_data":"bytes:(size=1029467)","dtype":"<class \'jax.numpy.float32\'>","event":"Initializing pipeline","level":"info","timestamp":"2023-08-28T00:32:56.452731Z"}'

$>
b'{"application":"whisper_stream","version":"0.0.1","python_version":"3.10.12","platform_architecture":["64bit",""],"model":"openai/whisper-tiny","mode":"whisper-jax-implicit-batching","event":"starting pre-compilation","level":"info","timestamp":"2023-08-28T00:33:02.274039Z"}'

$>
b'{"application":"whisper_stream","version":"0.0.1","python_version":"3.10.12","platform_architecture":["64bit",""],"model":"openai/whisper-tiny","mode":"whisper-jax-implicit-batching","time_taken":"10.31s","event":"finished pre-compilation","level":"info","timestamp":"2023-08-28T00:33:12.587567Z"}'
```

```python
# use
print(implicit_pipeline(data))
{'text': ' At the stroke of the midnight hour, when the world sleeps, India will await the light and freedom. A moment comes which comes but rarely in history. When we step out from the world to the new, when an agent'}
```

### Explicit Batching _(for Multiple Small Files)_

```python
# Warm-up
explicit_pipeline = pipeline_factory("jax","explicit_batching")(model, [data]*10)
```

```shell
$>
bb'{"application":"whisper_stream","version":"0.0.1","python_version":"3.10.12","platform_architecture":["64bit",""],"model":"openai/whisper-tiny","mode":"whisper-jax-explicit-batching","sample_data":"bytes array:(size=10)","dtype":"<class \'jax.numpy.float32\'>","task":"transcribe","return_timestamps":false,"max_new_tokens":25,"min_new_tokens":25,"event":"Initializing pipeline","level":"info","timestamp":"2023-08-28T00:35:38.456177Z"}'

$>
b'{"application":"whisper_stream","version":"0.0.1","python_version":"3.10.12","platform_architecture":["64bit",""],"model":"openai/whisper-tiny","mode":"whisper-jax-explicit-batching","event":"starting pre-compilation","level":"info","timestamp":"2023-08-28T00:35:38.458784Z"}'

$>
b'{"application":"whisper_stream","version":"0.0.1","python_version":"3.10.12","platform_architecture":["64bit",""],"model":"openai/whisper-tiny","mode":"whisper-jax-explicit-batching","time_taken":"17.05s","event":"finished pre-compilation","level":"info","timestamp":"2023-08-28T00:35:55.512401Z"}'
```

```python
# use
print(explicit_pipeline(data))
[' At the stroke of the midnight hour, when the world sleeps, India will await the light and freedom. A',
 ' At the stroke of the midnight hour, when the world sleeps, India will await the light and freedom. A',
 ' At the stroke of the midnight hour, when the world sleeps, India will await the light and freedom. A',
 ' At the stroke of the midnight hour, when the world sleeps, India will await the light and freedom. A',
 ' At the stroke of the midnight hour, when the world sleeps, India will await the light and freedom. A',
 ' At the stroke of the midnight hour, when the world sleeps, India will await the light and freedom. A',
 ' At the stroke of the midnight hour, when the world sleeps, India will await the light and freedom. A',
 ' At the stroke of the midnight hour, when the world sleeps, India will await the light and freedom. A',
 ' At the stroke of the midnight hour, when the world sleeps, India will await the light and freedom. A',
 ' At the stroke of the midnight hour, when the world sleeps, India will await the light and freedom. A']
```

## License

`whisper-stream` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
