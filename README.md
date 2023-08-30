[![CodeQL](https://github.com/ksquarekumar/whisper-stream/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ksquarekumar/whisper-stream/actions/workflows/github-code-scanning/codeql) [![Release](https://github.com/ksquarekumar/whisper-stream/actions/workflows/Release.yml/badge.svg)](https://github.com/ksquarekumar/whisper-stream/actions/workflows/Release.yml) ![Builds](https://codebuild.ap-south-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiS2FMcnRKSWhrNE0zYk0wR3dBRzlQSWVjQVBsbHhsYmwySWt6SG9zU1NRVWdrN1ZkTjJLNi83R1JPd3NWaDM5eU9sS0hVUUd4ODdUSGZ2Z3NCajZQbGNBPSIsIml2UGFyYW1ldGVyU3BlYyI6InFIYTNab2s1a3oxdWJVTnYiLCJtYXRlcmlhbFNldFNlcmlhbCI6Mn0%3D&branch=main)

# Whisper Stream

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

### From Git

```console
pip install "whisper-stream[data] @ git+https://github.com/ksquarekumar/whisper-stream.git@main"
```

### In Development Mode

- with `conda`

```console
conda install mamba
mamba env create -f conda.yml
mamba activate whisper_py310
pip install -e ."[data,benchmarks,dev,test]"
pre-commit install --install-hooks
```

- with `pyenv+conda`

```console
pyenv install mambaforge-22.9.0-3
pyenv shell mambaforge-22.9.0-3 && pyenv local mambaforge-22.9.0-3
mamba env create -f conda.yml
mamba activate whisper_py310
pip install -e ."[data,benchmarks,dev,test]"
pre-commit install --install-hooks
```

## [Whisper-JAX](./WhisperJax.md)

> `jax` modules partially vendored from [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax)

## Usage

### `JAX` Pipelines [Link](./notebooks/usage.ipynb)

## License

`whisper-stream` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
