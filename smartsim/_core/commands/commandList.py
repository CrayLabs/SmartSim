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

from collections.abc import MutableSequence
from .command import Command
import typing as t

class CommandList(MutableSequence):
    """Container for a seuence of commands
    """
    def __init__(self, commands: t.Optional[t.Union[Command, t.List[Command]]]):
        """CommandList constructor
        """
        self._commands: t.List[Command] = list(commands)

    @property
    def commands(self) -> t.List[Command]:
        """Get the command list.
        """
        return self._commands
    
    def __getitem__(self, idx: int) -> Command:
        """Get the command at the specified index.
        """
        return self._commands[idx]
    
    def __setitem__(self, idx: int, value: Command) -> None:
        """Set the command at the specified index.
        """
        self._commands[idx] = value

    def __delitem__(self, idx: int) -> None:
        """Delete the command at the specified index.
        """
        del self._commands[idx]

    def __len__(self) -> int:
        """Get the length of the command list.
        """
        return len(self._commands)

    def insert(self, idx: int, value: Command) -> None:
        """Insert a command at the specified index.
        """
        self._commands.insert(idx, value)