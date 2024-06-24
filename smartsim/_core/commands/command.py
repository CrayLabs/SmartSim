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

from ...settings.launchCommand import LauncherType


class Command(MutableSequence[str]):
    """Basic container for command information"""

    def __init__(self, launcher: LauncherType, command: t.List[str]) -> None:
        """Command constructor"""
        self._launcher = launcher
        self._command = command

    @property
    def launcher(self) -> LauncherType:
        """Get the launcher type.
        Return a reference to the LauncherType.
        """
        return self._launcher

    @property
    def command(self) -> t.List[str]:
        """Get the command list.
        Return a reference to the command list.
        """
        return self._command

    def __getitem__(self, idx: int) -> str:
        """Get the command at the specified index."""
        return self._command[idx]

    def __setitem__(self, idx: int, value: str) -> None:
        """Set the command at the specified index."""
        self._command[idx] = value

    def __delitem__(self, idx: int) -> None:
        """Delete the command at the specified index."""
        del self._command[idx]

    def __len__(self) -> int:
        """Get the length of the command list."""
        return len(self._command)

    def insert(self, idx: int, value: str) -> None:
        """Insert a command at the specified index."""
        self._command.insert(idx, value)

    def __str__(self) -> str:  # pragma: no cover
        string = f"\nLauncher: {self.launcher.value}\n"
        string += f"Command: {' '.join(str(cmd) for cmd in self.command)}"
        return string
