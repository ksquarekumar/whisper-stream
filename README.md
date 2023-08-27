# Whisper Stream

---

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
virtualenv ./.venv --system-site-packages
source ./.venv/bin/activate
pip install -e ."[data,dev,test]"
pre-commit install --install-hooks
```

## [Whisper-JAX](./WhisperJax.md)

> partially vendored from [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax)

## License

`whisper-stream` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
