# ruff: noqa: T201
# SPDX-FileCopyrightText: 2023-present krishnakumar <krishna.kumar@peak.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""Demonstrator for whisper on AWS"""
__version__ = "0.0.1"


def get_version() -> None:
    print(__version__)


__all__: list[str] = ["__version__", "get_version"]
