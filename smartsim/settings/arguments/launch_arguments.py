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

from __future__ import annotations

import copy
import textwrap
import typing as t
from abc import ABC, abstractmethod

from smartsim.log import get_logger

from ..._core.utils.helpers import fmt_dict

logger = get_logger(__name__)


class LaunchArguments(ABC):
    """Abstract base class for launcher arguments. It is the responsibility of
    child classes for each launcher to add methods to set input parameters and
    to maintain valid state between parameters set by a user.
    """

    def __init__(self, launch_args: t.Dict[str, str | None] | None) -> None:
        """Initialize a new `LaunchArguments` instance.

        :param launch_args: A mapping of arguments to (optional) values
        """
        self._launch_args = copy.deepcopy(launch_args) or {}
        """A dictionary of launch arguments"""

    @abstractmethod
    def launcher_str(self) -> str:
        """Get the string representation of the launcher"""

    @abstractmethod
    def set(self, arg: str, val: str | None) -> None:
        """Set a launch argument

        :param arg: The argument name to set
        :param val: The value to set the argument to as a `str` (if
            applicable). Otherwise `None`
        """

    def __str__(self) -> str:  # pragma: no-cover
        return textwrap.dedent(f"""\
            Launch Arguments:
                Launcher: {self.launcher_str()}
                Name: {type(self).__name__}
                Arguments:
                {fmt_dict(self._launch_args)}
            """)
