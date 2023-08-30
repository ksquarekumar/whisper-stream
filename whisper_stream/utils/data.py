#!python3
# SPDX-FileCopyrightText: 2023-present krishnakumar <krishna.kumar@peak.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Literal, overload

from itertools import chain

from whisper_stream.constants import DEFAULT_DL_PATH
from whisper_stream.utils.parallel import DEFAULT_PARALLEL_BACKEND, delayed


def _read_file(file: Path) -> bytes:
    with file.absolute():
        return file.read_bytes()


@overload
def load_data_samples_from_path(
    sample_file_glob_or_globs: str | list[str],
    directory: Path = DEFAULT_DL_PATH,
    *,
    return_all: Literal[False] = False,
    binary_mode: Literal[False] = False,
) -> str:
    ...


@overload
def load_data_samples_from_path(
    sample_file_glob_or_globs: str | list[str],
    directory: Path = DEFAULT_DL_PATH,
    *,
    return_all: Literal[False] = False,
    binary_mode: Literal[True],
) -> bytes:
    ...


@overload
def load_data_samples_from_path(
    sample_file_glob_or_globs: str | list[str],
    directory: Path = DEFAULT_DL_PATH,
    *,
    return_all: Literal[True],
    binary_mode: Literal[False] = False,
) -> list[str]:
    ...


@overload
def load_data_samples_from_path(
    sample_file_glob_or_globs: str | list[str],
    directory: Path = DEFAULT_DL_PATH,
    *,
    return_all: Literal[True],
    binary_mode: Literal[True],
) -> list[bytes]:
    ...


def load_data_samples_from_path(
    sample_file_glob_or_globs: str | list[str],
    directory: Path = DEFAULT_DL_PATH,
    *,
    return_all: Literal[True, False] = False,
    binary_mode: Literal[True, False] = False,
) -> str | bytes | list[str] | list[bytes]:
    """load data `file-name/(s): (Path | list[Path]` and or `file/(s) (bytes | list[bytes])` from a `directory` given a `glob` pattern.

    Args:
        sample_file_glob_or_globs (str | list[str]):
            a single glob-patter or a list of glob(s),
            this will be matched against the directory to find and load files.
        directory (Path, optional):
            the directory path where files will be seached for
            Defaults to `whisper_stream.constants.DEFAULT_DL_PATH`.
        return_all (Literal[True, False], optional):
            whether or not to stop returning beyond the first match, Defaults to False.
            return_all will read and return as many fil
        binary_mode (Literal[True, False], optional): _description_. Defaults to False.
            whether to return file path(s) or their contents as bytes
            setting this to true will return the file instead as `bytes`
    Raises:
        FileNotFoundError: if globs result in no match

    Returns:
        str | bytes | list[str] | list[bytes]:
        - a single file's `Path` as `str` when `(return_all:False, binary_mode: False)`
        - a singe file as `bytes` when `(return_all:False, , binary_mode: True)`
        - list of file paths when `(return_all:True, , binary_mode: False)`
        - list of file as `bytes` when `(return_all:True, , binary_mode: True)`
    """
    files: list[Path] = (
        list(directory.glob(f"./{sample_file_glob_or_globs}"))
        if isinstance(sample_file_glob_or_globs, str)
        else list(set(chain(*(directory.glob(f"./{glob}") for glob in sample_file_glob_or_globs))))
    )
    if len(files) >= 1:
        if return_all == False:
            if binary_mode == False:
                return str(files[0].absolute())
            return _read_file(files[0])
        if binary_mode == False:
            return list(str(file.absolute()) for file in files)
        return list(DEFAULT_PARALLEL_BACKEND(delayed(_read_file)(file) for file in files))
    msg: str = f"File with glob {sample_file_glob_or_globs} do not exist in {directory.absolute()}"
    raise FileNotFoundError(msg)


__all__: list[str] = ["load_data_samples_from_path"]
