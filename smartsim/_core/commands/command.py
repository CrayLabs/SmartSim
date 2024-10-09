# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import typing as t
from collections.abc import MutableSequence
from copy import deepcopy

from typing_extensions import Self


class Command(MutableSequence[str]):
    """Basic container for command information"""

    def __init__(self, command: t.List[str]) -> None:
        if not command:
            raise TypeError("Command list cannot be empty")
        if not all(isinstance(item, str) for item in command):
            raise TypeError("All items in the command list must be strings")
        """Command constructor"""
        self._command = command

    @property
    def command(self) -> t.List[str]:
        """Get the command list.
        Return a reference to the command list.
        """
        return self._command

    @t.overload
    def __getitem__(self, idx: int) -> str: ...
    @t.overload
    def __getitem__(self, idx: slice) -> Self: ...
    def __getitem__(self, idx: t.Union[int, slice]) -> t.Union[str, Self]:
        """Get the command at the specified index."""
        cmd = self._command[idx]
        if isinstance(cmd, str):
            return cmd
        return type(self)(cmd)

    @t.overload
    def __setitem__(self, idx: int, value: str) -> None: ...
    @t.overload
    def __setitem__(self, idx: slice, value: t.Iterable[str]) -> None: ...
    def __setitem__(
        self, idx: t.Union[int, slice], value: t.Union[str, t.Iterable[str]]
    ) -> None:
        """Set the command at the specified index."""
        if isinstance(idx, int):
            if not isinstance(value, str):
                raise TypeError(
                    "Value must be of type `str` when assigning to an index"
                )
            self._command[idx] = deepcopy(value)
            return
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise TypeError("Value must be a list of strings when assigning to a slice")
        self._command[idx] = (deepcopy(val) for val in value)

    def __delitem__(self, idx: t.Union[int, slice]) -> None:
        """Delete the command at the specified index."""
        del self._command[idx]

    def __len__(self) -> int:
        """Get the length of the command list."""
        return len(self._command)

    def insert(self, idx: int, value: str) -> None:
        """Insert a command at the specified index."""
        self._command.insert(idx, value)

    def __str__(self) -> str:  # pragma: no cover
        string = f"\nCommand: {' '.join(str(cmd) for cmd in self.command)}"
        return string
