[![CodeQL](https://github.com/ksquarekumar/whisper-stream/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ksquarekumar/whisper-stream/actions/workflows/github-code-scanning/codeql)

# _Whisper-Stream_ 🌬️

> This project aims to provide applications/pipelines and common code for performing _fast_ `automatic speech recognition`, `transcription` and `translation` using open-source models based on _[open-ai](https://openai.com/)'s_ [whisper](https://openai.com/research/whisper) project.

_**Table of Contents**_

- [Installation](#installation)
- [Development](#development)
- [Features](#features)
- [License](#license)

## _Installation_

### _Consuming this project via `pip`_

```shell
pip install "whisper-stream[{feature},...] @ git+https://github.com/ksquarekumar/whisper-stream.git@main"
```

## _Development_

> This project uses [pyenv](https://github.com/pyenv/pyenv), [mamba](https://github.com/mamba-org/mamba) and [`poetry`](https://python-poetry.org/) to manage environments, dependencies and building wheels.

> For available extras/features refer to the `extras` section under `[tool.poetry.extras]` project [manifest](https://github.com/ksquarekumar/whisper-stream/blob/main/pyproject.toml)

### _Step by Step installation._

#### _1. Clone this repo._

```
git clone git+https://github.com/ksquarekumar/whisper-stream.git
```

##### _1.1. Install `pyenv`._

```shell
curl https://pyenv.run | bash
```

#### _2. Install a `mambaforge` environment with `pyenv`._

```shell
pyenv install mambaforge-22.9.0-3 && pyenv shell mambaforge-22.9.0-3 && mamba activate base
mamba install conda-lock poetry
mamba update --name base --update-all
```

##### _2.1. Optionally, set it as the default global interpreter in `pyenv`._

```shell
pyenv global mambaforge-22.9.0-3
exec $(SHELL)
```

#### _3. Create a project environment **(named: `whisper_py311`)** from the existing [`conda.yml`](https://github.com/ksquarekumar/whisper-stream/blob/main/conda.yml) manifest._

```shell
mamba env create -f conda.yml && mamba activate whisper_py311
```

##### _3.1. Optionally, set it as the default local environment for this project._

```shell
pyenv local mambaforge-22.9.0-3/envs/whisper_py311
exec $(SHELL)
```

#### _4. Install project dependencies in a project local virtual environment with `poetry`._

```shell
mamba activate whisper_py311
pip install conda-lock poetry-conda
poetry install -E "[list of features,..]"
```

- For `development` you probably want all of `"[dev,test]"` groups so `poetry install` is what you need

- For `non-development` install you probably want to exclude `[dev,test]` groups, so install with:

```shell
poetry install --only main
```

#### _5. Optional, setup project local commit and git hooks_

```shell
pre-commit install --install-hooks
```

### _TL;DR Version for `CI` Builds_

> assumes `source` is present in system

- within the `system` python for containers

```console
pip install requirements.{feature}.txt
pip install .["feature"]
```

- with `conda` as the system's environment manager

- install `pyenv`

```console
conda install mamba
mamba env update -f conda.yml
pip install requirements.{feature}.txt
pip install .["feature"]
```

### Features

#### [Service-Jax](./docs/ServiceJax.md)

## License

> some `jax` modules are partially vendored from [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax)

`whisper-stream` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
