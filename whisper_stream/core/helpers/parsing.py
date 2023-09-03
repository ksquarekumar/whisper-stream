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
import inspect
from types import MappingProxyType
from typing import Any, Literal, Type, TypeGuard, Callable


def is_bytes(x: Any | None) -> TypeGuard[bytes]:
    if x is not None and isinstance(x, bytes):
        return True
    return False


def is_bytes_array(x: Any | None) -> TypeGuard[list[bytes]]:
    if x is not None and all(isinstance(y, bytes) for y in x):
        return True
    return False


def parse_known_kwargs(
    func_or_class: Callable[..., Any] | Type[Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    signature: inspect.Signature = inspect.signature(func_or_class)
    known_params: MappingProxyType[str, inspect.Parameter] = signature.parameters

    if inspect.isclass(func_or_class):
        # For classes, remove the first parameter ("self") from the known parameters
        known_params = MappingProxyType(
            {name: param for name, param in known_params.items() if name != "self"}
        )

    _safe_kwargs: dict[str, Any] = {}
    for kwarg, value in kwargs.items():
        if kwarg in known_params:
            _safe_kwargs[kwarg] = value

    return _safe_kwargs


LanguageIDs = Literal[
    "<|af|>",
    "<|am|>",
    "<|ar|>",
    "<|as|>",
    "<|az|>",
    "<|ba|>",
    "<|be|>",
    "<|bg|>",
    "<|bn|>",
    "<|bo|>",
    "<|br|>",
    "<|bs|>",
    "<|ca|>",
    "<|cs|>",
    "<|cy|>",
    "<|da|>",
    "<|de|>",
    "<|el|>",
    "<|en|>",
    "<|es|>",
    "<|et|>",
    "<|eu|>",
    "<|fa|>",
    "<|fi|>",
    "<|fo|>",
    "<|fr|>",
    "<|gl|>",
    "<|gu|>",
    "<|haw|>",
    "<|ha|>",
    "<|he|>",
    "<|hi|>",
    "<|hr|>",
    "<|ht|>",
    "<|hu|>",
    "<|hy|>",
    "<|id|>",
    "<|is|>",
    "<|it|>",
    "<|ja|>",
    "<|jw|>",
    "<|ka|>",
    "<|kk|>",
    "<|km|>",
    "<|kn|>",
    "<|ko|>",
    "<|la|>",
    "<|lb|>",
    "<|ln|>",
    "<|lo|>",
    "<|lt|>",
    "<|lv|>",
    "<|mg|>",
    "<|mi|>",
    "<|mk|>",
    "<|ml|>",
    "<|mn|>",
    "<|mr|>",
    "<|ms|>",
    "<|mt|>",
    "<|my|>",
    "<|ne|>",
    "<|nl|>",
    "<|nn|>",
    "<|no|>",
    "<|oc|>",
    "<|pa|>",
    "<|pl|>",
    "<|ps|>",
    "<|pt|>",
    "<|ro|>",
    "<|ru|>",
    "<|sa|>",
    "<|sd|>",
    "<|si|>",
    "<|sk|>",
    "<|sl|>",
    "<|sn|>",
    "<|so|>",
    "<|sq|>",
    "<|sr|>",
    "<|su|>",
    "<|sv|>",
    "<|sw|>",
    "<|ta|>",
    "<|te|>",
    "<|tg|>",
    "<|th|>",
    "<|tk|>",
    "<|tl|>",
    "<|tr|>",
    "<|tt|>",
    "<|uk|>",
    "<|ur|>",
    "<|uz|>",
    "<|vi|>",
    "<|yi|>",
    "<|yo|>",
    "<|zh|>",
]


__all__: list[str] = ["is_bytes", "is_bytes_array", "parse_known_kwargs", "LanguageIDs"]
