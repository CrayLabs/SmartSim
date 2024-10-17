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

from .command import Command


class CommandList(MutableSequence[Command]):
    """Container for a Sequence of Command objects"""

    def __init__(self, commands: t.Optional[t.Union[Command, t.List[Command]]] = None):
        """CommandList constructor"""
        if commands is None:
            commands = []
        elif isinstance(commands, Command):
            commands = [commands]
        self._commands: t.List[Command] = list(commands)

    @property
    def commands(self) -> t.List[Command]:
        """Get the Command list.
        Return a reference to the Command list.
        """
        return self._commands

    @t.overload
    def __getitem__(self, idx: int) -> Command: ...
    @t.overload
    def __getitem__(self, idx: slice) -> t.List[Command]: ...
    def __getitem__(
        self, idx: t.Union[slice, int]
    ) -> t.Union[Command, t.List[Command]]:
        """Get the Command at the specified index."""
        return self._commands[idx]

    @t.overload
    def __setitem__(self, idx: int, value: Command) -> None: ...
    @t.overload
    def __setitem__(self, idx: slice, value: t.Iterable[Command]) -> None: ...
    def __setitem__(
        self, idx: t.Union[int, slice], value: t.Union[Command, t.Iterable[Command]]
    ) -> None:
        """Set the Commands at the specified index."""
        if isinstance(idx, int):
            if not isinstance(value, Command):
                raise ValueError(
                    "Value must be of type `Command` when assigning to an index"
                )
            self._commands[idx] = deepcopy(value)
            return
        if not isinstance(value, list):
            raise ValueError(
                "Value must be a list of Commands when assigning to a slice"
            )
        for sublist in value:
            if not isinstance(sublist.command, list) or not all(
                isinstance(item, str) for item in sublist.command
            ):
                raise ValueError(
                    "Value sublists must be a list of Commands when assigning to a slice"
                )
        self._commands[idx] = (deepcopy(val) for val in value)

    def __delitem__(self, idx: t.Union[int, slice]) -> None:
        """Delete the Command at the specified index."""
        del self._commands[idx]

    def __len__(self) -> int:
        """Get the length of the Command list."""
        return len(self._commands)

    def insert(self, idx: int, value: Command) -> None:
        """Insert a Command at the specified index."""
        self._commands.insert(idx, value)

    def __str__(self) -> str:  # pragma: no cover
        string = "\n\nCommand List:\n\n"
        for counter, cmd in enumerate(self.commands):
            string += f"CommandList index {counter} value:"
            string += f"{cmd}\n\n"
        return string
