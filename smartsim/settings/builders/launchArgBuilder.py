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
import typing as t
from abc import ABC, abstractmethod

from smartsim.log import get_logger

from ..._core.utils.helpers import fmt_dict

logger = get_logger(__name__)


_T = t.TypeVar("_T")


class LaunchArgBuilder(ABC, t.Generic[_T]):
    """Abstract base class that defines all generic launcher
    argument methods that are not supported.  It is the
    responsibility of child classes for each launcher to translate
    the input parameter to a properly formatted launcher argument.
    """

    def __init__(self, launch_args: t.Dict[str, str | None] | None) -> None:
        self._launch_args = copy.deepcopy(launch_args) or {}

    @abstractmethod
    def launcher_str(self) -> str:
        """Get the string representation of the launcher"""

    @abstractmethod
    def set(self, arg: str, val: str | None) -> None:
        """Set the launch arguments"""

    @abstractmethod
    def finalize(self, exe: ExecutableLike, env: t.Mapping[str, str | None], job_execution_path: str) -> t.Tuple[t.Sequence[str], str]:
        """Prepare an entity for launch using the built options"""

    def format_launch_args(self) -> t.Union[t.List[str], None]:
        """Build formatted launch arguments"""
        logger.warning(
            f"format_launcher_args() not supported for {self.launcher_str()}."
        )
        return None

    def format_comma_sep_env_vars(
        self, env_vars: t.Dict[str, t.Optional[str]]
    ) -> t.Union[t.Tuple[str, t.List[str]], None]:
        """Build environment variable string for Slurm
        Slurm takes exports in comma separated lists
        the list starts with all as to not disturb the rest of the environment
        for more information on this, see the slurm documentation for srun
        :returns: the formatted string of environment variables
        """
        logger.warning(
            f"format_comma_sep_env_vars() not supported for {self.launcher_str()}."
        )
        return None

    def format_env_vars(
        self, env_vars: t.Dict[str, t.Optional[str]]
    ) -> t.Union[t.List[str], None]:
        """Build bash compatible environment variable string for Slurm
        :returns: the formatted string of environment variables
        """
        logger.warning(f"format_env_vars() not supported for {self.launcher_str()}.")
        return None

    def __str__(self) -> str:  # pragma: no-cover
        string = f"\nLaunch Arguments:\n{fmt_dict(self._launch_args)}"
        return string


class ExecutableLike(t.Protocol):
    @abstractmethod
    def as_program_arguments(self) -> t.Sequence[str]: ...
