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

from smartsim.log import get_logger

from .._core.utils.helpers import fmt_dict
from .baseSettings import BaseSettings
from .builders import LaunchArgBuilder
from .builders.launch.alps import AprunArgBuilder
from .builders.launch.dragon import DragonArgBuilder
from .builders.launch.local import LocalArgBuilder
from .builders.launch.lsf import JsrunArgBuilder
from .builders.launch.mpi import MpiArgBuilder, MpiexecArgBuilder, OrteArgBuilder
from .builders.launch.pals import PalsMpiexecArgBuilder
from .builders.launch.slurm import SlurmArgBuilder
from .common import StringArgument
from .launchCommand import LauncherType

logger = get_logger(__name__)


class LaunchSettings(BaseSettings):
    def __init__(
        self,
        launcher: t.Union[LauncherType, str],
        launch_args: StringArgument | None = None,
        env_vars: StringArgument | None = None,
    ) -> None:
        try:
            self._launcher = LauncherType(launcher)
        except ValueError:
            raise ValueError(f"Invalid launcher type: {launcher}")
        self._arg_builder = self._get_arg_builder(launch_args)
        self.env_vars = copy.deepcopy(env_vars) if env_vars else {}

    @property
    def launcher(self) -> str:
        """Return the launcher name."""
        return self._launcher.value

    @property
    def launch_args(self) -> LaunchArgBuilder:
        """Return the launch argument translator."""
        return self._arg_builder

    @launch_args.setter
    def launch_args(self, args: t.Mapping[str, str]) -> None:
        """Update the launch arguments."""
        self.launch_args._launch_args.clear()
        for k, v in args.items():
            self.launch_args.set(k, v)

    @property
    def env_vars(self) -> t.Mapping[str, str | None]:
        """Return an immutable list of attached environment variables."""
        return copy.deepcopy(self._env_vars)

    @env_vars.setter
    def env_vars(self, value: t.Dict[str, str]) -> None:
        """Set the environment variables."""
        self._env_vars = copy.deepcopy(value)

    def _get_arg_builder(self, launch_args: StringArgument | None) -> LaunchArgBuilder:
        """Map the Launcher to the LaunchArgBuilder"""
        if self._launcher == LauncherType.Slurm:
            return SlurmArgBuilder(launch_args)
        elif self._launcher == LauncherType.Mpiexec:
            return MpiexecArgBuilder(launch_args)
        elif self._launcher == LauncherType.Mpirun:
            return MpiArgBuilder(launch_args)
        elif self._launcher == LauncherType.Orterun:
            return OrteArgBuilder(launch_args)
        elif self._launcher == LauncherType.Alps:
            return AprunArgBuilder(launch_args)
        elif self._launcher == LauncherType.Lsf:
            return JsrunArgBuilder(launch_args)
        elif self._launcher == LauncherType.Pals:
            return PalsMpiexecArgBuilder(launch_args)
        elif self._launcher == LauncherType.Dragon:
            return DragonArgBuilder(launch_args)
        elif self._launcher == LauncherType.Local:
            return LocalArgBuilder(launch_args)
        else:
            raise ValueError(f"Invalid launcher type: {self._launcher}")

    def update_env(self, env_vars: t.Dict[str, str | None]) -> None:
        """Update the job environment variables

        To fully inherit the current user environment, add the
        workload-manager-specific flag to the launch command through the
        :meth:`add_exe_args` method. For example, ``--export=ALL`` for
        slurm, or ``-V`` for PBS/aprun.


        :param env_vars: environment variables to update or add
        :raises TypeError: if env_vars values cannot be coerced to strings
        """
        # Coerce env_vars values to str as a convenience to user
        for env, val in env_vars.items():
            if not isinstance(env, str):
                raise TypeError(f"The key '{env}' of env_vars should be of type str")
            if not isinstance(val, (str, type(None))):
                raise TypeError(
                    f"The value '{val}' of env_vars should be of type str or None"
                )
        self._env_vars.update(env_vars)

    def format_env_vars(self) -> t.Union[t.List[str], None]:
        """Build bash compatible environment variable string for Slurm
        :returns: the formatted string of environment variables
        """
        return self._arg_builder.format_env_vars(self._env_vars)

    def format_comma_sep_env_vars(self) -> t.Union[t.Tuple[str, t.List[str]], None]:
        """Build environment variable string for Slurm
        Slurm takes exports in comma separated lists
        the list starts with all as to not disturb the rest of the environment
        for more information on this, see the slurm documentation for srun
        :returns: the formatted string of environment variables
        """
        return self._arg_builder.format_comma_sep_env_vars(self._env_vars)

    def format_launch_args(self) -> t.Union[t.List[str], None]:
        """Return formatted launch arguments
        For ``RunSettings``, the run arguments are passed
        literally with no formatting.
        :return: list run arguments for these settings
        """
        return self._arg_builder.format_launch_args()

    def __str__(self) -> str:  # pragma: no-cover
        string = f"\nLauncher: {self.launcher}"
        if self.launch_args:
            string += str(self.launch_args)
        if self.env_vars:
            string += f"\nEnvironment variables: \n{fmt_dict(self.env_vars)}"
        return string
