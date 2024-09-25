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
from .base_settings import BaseSettings
from .common import StringArgument
from .launch_command import LauncherType

logger = get_logger(__name__)


class LaunchSettings(BaseSettings):
    """The LaunchSettings class stores launcher configuration settings and is
    used to inject launcher-specific behavior into a job.

    LaunchSettings is designed to be extended by a LaunchArguments child class that
    corresponds to the launcher provided during initialization. The supported launchers
    are Dragon, Slurm, PALS, ALPS, Local, Mpiexec, Mpirun, Orterun, and LSF. Using the
    LaunchSettings class, users can:

    - Set the launcher type of a job.
    - Configure launch arguments and environment variables.
    - Access and modify custom launch arguments.
    - Update environment variables.
    - Retrieve information associated with the ``LaunchSettings`` object.
        - The launcher value (LaunchSettings.launcher).
        - The derived LaunchSettings child class (LaunchSettings.launch_args).
        - The set environment variables (LaunchSettings.env_vars).
    """

    def __init__(
        self,
        launcher: t.Union[LauncherType, str],
        launch_args: StringArgument | None = None,
        env_vars: StringArgument | None = None,
    ) -> None:
        """Initialize a LaunchSettings instance.

        The "launcher" of SmartSim LaunchSettings will determine the
        child type assigned to the LaunchSettings.launch_args attribute.
        For example, to configure a job for SLURM, assign LaunchSettings.launcher
        to "slurm" or LauncherType.Slurm:

        .. highlight:: python
        .. code-block:: python

            srun_settings = LaunchSettings(launcher="slurm")
            # OR
            srun_settings = LaunchSettings(launcher=LauncherType.Slurm)

        This will assign a SlurmLaunchArguments object to ``srun_settings.launch_args``.
        Using the object, users may access the child class functions to set
        batch configurations. For example:

        .. highlight:: python
        .. code-block:: python

            srun_settings.launch_args.set_nodes(5)
            srun_settings.launch_args.set_cpus_per_task(2)

        To set customized launch arguments, use the  `set()`function provided by
        the LaunchSettings child class. For example:

        .. highlight:: python
        .. code-block:: python

            srun_settings.launch_args.set(key="nodes", value="6")

        If the key already exists in the existing launch arguments, the value will
        be overwritten.

        :param launcher: The type of launcher to initialize (e.g., Dragon, Slurm,
            PALS, ALPS, Local, Mpiexec, Mpirun, Orterun, LSF)
        :param launch_args: A dictionary of arguments for the launcher, where the keys
            are strings and the values can be either strings or None. This argument is optional
            and defaults to None.
        :param env_vars: Environment variables for the launch settings, where the keys
            are strings and the values can be either strings or None. This argument is
            also optional and defaults to None.
        :raises ValueError: Raises if the launcher provided does not exist.
        """
        try:
            self._launcher = LauncherType(launcher)
            """The launcher type"""
        except ValueError:
            raise ValueError(f"Invalid launcher type: {launcher}")
        self._arguments = self._get_arguments(launch_args)
        """The LaunchSettings child class based on launcher type"""
        self.env_vars = env_vars or {}
        """The environment configuration"""

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
        return self._env_vars

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
        :raises ValueError: An invalid launcher type was provided.
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
