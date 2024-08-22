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
from .arguments import LaunchArguments
from .arguments.launch.alps import AprunLaunchArguments
from .arguments.launch.dragon import DragonLaunchArguments
from .arguments.launch.local import LocalLaunchArguments
from .arguments.launch.lsf import JsrunLaunchArguments
from .arguments.launch.mpi import (
    MpiexecLaunchArguments,
    MpirunLaunchArguments,
    OrterunLaunchArguments,
)
from .arguments.launch.pals import PalsMpiexecLaunchArguments
from .arguments.launch.slurm import SlurmLaunchArguments
from .baseSettings import BaseSettings
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
        self._arguments = self._get_arguments(launch_args)
        self.env_vars = env_vars or {}

    @property
    def launcher(self) -> str:
        """The launcher type

        :returns: The launcher type's string representation
        """
        return self._launcher.value

    @property
    def launch_args(self) -> LaunchArguments:
        """The launch argument

        :returns: The launch arguments
        """
        return self._arguments

    @property
    def env_vars(self) -> t.Mapping[str, str | None]:
        """A mapping of environment variables to set or remove. This mapping is
        a deep copy of the mapping used by the settings and as such altering
        will not mutate the settings.

        :returns: An environment mapping
        """
        return copy.deepcopy(self._env_vars)

    @env_vars.setter
    def env_vars(self, value: dict[str, str | None]) -> None:
        """Set the environment variables to a new mapping. This setter will
        make a copy of the mapping and as such altering the original mapping
        will not mutate the settings.

        :param value: The new environment mapping
        """
        self._env_vars = copy.deepcopy(value)

    def _get_arguments(self, launch_args: StringArgument | None) -> LaunchArguments:
        """Map the Launcher to the LaunchArguments. This method should only be
        called once during construction.

        :param launch_args: A mapping of arguments names to values to be used
            to initialize the arguments
        :returns: The appropriate type for the settings instance.
        """
        if self._launcher == LauncherType.Slurm:
            return SlurmLaunchArguments(launch_args)
        elif self._launcher == LauncherType.Mpiexec:
            return MpiexecLaunchArguments(launch_args)
        elif self._launcher == LauncherType.Mpirun:
            return MpirunLaunchArguments(launch_args)
        elif self._launcher == LauncherType.Orterun:
            return OrterunLaunchArguments(launch_args)
        elif self._launcher == LauncherType.Alps:
            return AprunLaunchArguments(launch_args)
        elif self._launcher == LauncherType.Lsf:
            return JsrunLaunchArguments(launch_args)
        elif self._launcher == LauncherType.Pals:
            return PalsMpiexecLaunchArguments(launch_args)
        elif self._launcher == LauncherType.Dragon:
            return DragonLaunchArguments(launch_args)
        elif self._launcher == LauncherType.Local:
            return LocalLaunchArguments(launch_args)
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

    def __str__(self) -> str:  # pragma: no-cover
        string = f"\nLauncher: {self.launcher}{self.launch_args}"
        if self.env_vars:
            string += f"\nEnvironment variables: \n{fmt_dict(self.env_vars)}"
        return string
