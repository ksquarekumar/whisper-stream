#
# # Copyright © 2023 krishnakumar <ksquarekumar@gmail.com>.
# #
# # Licensed under the Apache License, Version 2.0 (the "License"). You
# # may not use this file except in compliance with the License. A copy of
# # the License is located at:
# #
# # https://github.com/ksquarekumar/whisper-stream/blob/main/LICENSE
# #
# # or in the "license" file accompanying this file. This file is
# # distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# # ANY KIND, either express or implied. See the License for the specific
# # language governing permissions and limitations under the License.
# #
# # This file is part of the whisper-stream.
# # see (https://github.com/ksquarekumar/whisper-stream)
# #
# # SPDX-License-Identifier: Apache-2.0
# #
# # You should have received a copy of the APACHE LICENSE, VERSION 2.0
# # along with this program. If not, see <https://apache.org/licenses/LICENSE-2.0>
#
from pathlib import Path
from typing import Any, Final, TypeAlias

import jax.numpy as jnp

from jax._src.numpy.lax_numpy import _ScalarMeta

from whisper_stream.core.constants import (
    CACHE_PATH_PREFIX,
    create_package_directories_if_not_exists,
)
from whisper_stream.core.helpers.parallel import get_backend, delayed, Parallel
from whisper_stream.core.logger import BoundLogger, get_application_logger

from transformers.models.whisper.tokenization_whisper_fast import WhisperTokenizerFast
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)

JAX_CACHE_PATH: Final[Path] = CACHE_PATH_PREFIX / "jax"

JAXScalarDType: TypeAlias = _ScalarMeta

JAXValidDtypesMapping: Final[dict[str, JAXScalarDType]] = {
    "FLOAT32": jnp.float32,
    "BFLOAT16": jnp.bfloat16,
    "FLOAT16": jnp.float16,
}

JAXDecoderIDType = list[tuple[int, str | Any]]

JAXWhisperTokenizers: TypeAlias = WhisperTokenizerFast | WhisperTokenizer

JAXThreadParallel: Parallel = get_backend("threading")
JAXMPParallel: Parallel = get_backend("loky")

JaxShardingRulesetType = tuple[tuple[str, str | None], ...]
FLAX_DEFAULT_DP_LOGICAL_AXES: JaxShardingRulesetType = (
    ("batch", "data"),
    ("mlp", None),
    ("heads", None),
    ("vocab", None),
    ("embed", None),
    ("embed", None),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None),
)

# import side-effects
create_package_directories_if_not_exists(
    scope="whisper-steam:jax-pipelines:init",
    path_or_paths=JAX_CACHE_PATH,
)
# side-effects done

__all__: list[str] = [
    "BoundLogger",
    "get_backend",
    "delayed",
    "Parallel",
    "get_application_logger",
    "JAX_CACHE_PATH",
    "JAXScalarDType",
    "JAXValidDtypesMapping",
    "JAXDecoderIDType",
    "JAXWhisperTokenizers",
    "JAXThreadParallel",
    "JAXMPParallel",
    "JaxShardingRulesetType",
    "FLAX_DEFAULT_DP_LOGICAL_AXES",
    "WhisperTokenizerFast",
    "WhisperTokenizer",
    "WhisperFeatureExtractor",
]
