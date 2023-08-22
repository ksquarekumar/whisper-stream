# Whisper Stream

---

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

### From Git

```console
pip install "whisper-stream @ git+https://github.com/ksquarekumar/whisper-stream.git"
```

### In Development Mode

```console
virtualenv ./.venv
source ./.venv/bin/activate
pip install --upgrade pip wheel build sdist flit
pip install -e ."[dev, test]"
pre-commit install --install-hooks && pre-commit autoupdate
```

## License

`whisper-stream` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
