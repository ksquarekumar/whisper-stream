# SPDX-FileCopyrightText: 2023-present krishnakumar <krishna.kumar@peak.ai>
#
# SPDX-License-Identifier: Apache-2.0
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


def parse_known_kwargs(func_or_class: Callable[..., Any] | Type[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature: inspect.Signature = inspect.signature(func_or_class)
    known_params: MappingProxyType[str, inspect.Parameter] = signature.parameters

    if inspect.isclass(func_or_class):
        # For classes, remove the first parameter ("self") from the known parameters
        known_params = MappingProxyType({name: param for name, param in known_params.items() if name != "self"})

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
