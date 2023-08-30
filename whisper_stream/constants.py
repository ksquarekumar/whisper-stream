# SPDX-FileCopyrightText: 2023-present krishnakumar <krishna.kumar@peak.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Final

JAX_CACHE_PATH: Final[str] = str(Path((Path(__file__).parent.parent / ".jax_cache")).absolute())

DEFAULT_DL_PATH: Path = Path(Path(__file__).parent.parent / "data").absolute()

# side-effects
Path(JAX_CACHE_PATH).mkdir(exist_ok=True)

__all__: list[str] = ["JAX_CACHE_PATH", "DEFAULT_DL_PATH"]
