# ruff: noqa: T201
# SPDX-FileCopyrightText: 2023-present krishnakumar <krishna.kumar@peak.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
NOTICE for [whisper-stream.vendored](https://github.com/ksquarekumar/whisper-stream/vendored)

*   This project includes code from the following third-party sources, which are subject
    to the terms and conditions of the Apache License, Version 2.0:

## List of Projects

1. [whisper-jax](<(https://github.com/sanchit-gandhi/whisper-jax)>)

   1.1. Description

   ```
   This repository contains optimised JAX code for OpenAI's Whisper Model,
   largely built on the 🤗 Hugging Face Transformers Whisper implementation.
   ```

   1.2. Copyright [2023-] [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax) [contributors](https://github.com/sanchit-gandhi/whisper-jax/graphs/contributors).

   1.3. This product includes software developed at [whisper-jax](https://github.com/sanchit-gandhi/whisper-jax).

   1.4. More information about this third-party project can be found at [whisper-jax/homepage](https://github.com/sanchit-gandhi/whisper-jax).

   1.5. The full text of the Apache License, Version 2.0, is available at https://www.apache.org/licenses/LICENSE-2.0.

## ...

 Portions of this software are also distributed with their own licenses, which can be found in the respective source files and directories.

 End of NOTICE for [whisper-stream](https://github.com/ksquarekumar/whisper-stream/vendored)
"""
from whisper_stream.vendored import whisper_jax
from whisper_stream.vendored import benchmarks as whisper_jax_benchmarks

__all__: list[str] = ["whisper_jax", "whisper_jax_benchmarks"]
