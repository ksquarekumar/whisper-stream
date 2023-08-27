import inspect
from types import MappingProxyType
from typing import Any, Type, TypeGuard, Callable


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
        # For classes, remove the first parameter ('self') from the known parameters
        known_params = MappingProxyType({name: param for name, param in known_params.items() if name != 'self'})

    _safe_kwargs: dict[str, Any] = {}
    for kwarg, value in kwargs.items():
        if kwarg in known_params:
            _safe_kwargs[kwarg] = value

    return _safe_kwargs


__all__: list[str] = ["is_bytes", "is_bytes_array", "parse_known_kwargs"]
