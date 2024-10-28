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

import typing as t
from copy import deepcopy

from smartsim._core.commands.launch_commands import LaunchCommands
from smartsim._core.utils.helpers import check_name
from smartsim.launchable.base_job import BaseJob
from smartsim.log import get_logger
from smartsim.settings import LaunchSettings

logger = get_logger(__name__)

if t.TYPE_CHECKING:
    from smartsim.entity.entity import SmartSimEntity


@t.final
class Job(BaseJob):
    """A Job holds a reference to a SmartSimEntity and associated
    LaunchSettings prior to launch. It is responsible for turning
    the stored SmartSimEntity and LaunchSettings into commands that can be
    executed by a launcher. Jobs are designed to be started by the Experiment.
    """

    def __init__(
        self,
        entity: SmartSimEntity,
        launch_settings: LaunchSettings,
        name: str | None = None,
    ):
        """Initialize a ``Job``

        Jobs require a SmartSimEntity and a LaunchSettings. Optionally, users may provide
        a name. To create a simple Job that echos `Hello World!`, consider the example below:

        .. highlight:: python
        .. code-block:: python

            # Create an application that runs the 'echo' command
            my_app = Application(name="my_app", exe="echo", exe_args="Hello World!")
            # Define the launch settings using SLURM
            srun_settings = LaunchSettings(launcher="slurm")

            # Create a Job with the `my_app` and `srun_settings`
            my_job = Job(my_app, srun_settings, name="my_job")

        :param entity: the SmartSimEntity object
        :param launch_settings: the LaunchSettings object
        :param name: the Job name
        """
        super().__init__()
        """Initialize the parent class BaseJob"""
        self.entity = entity
        """Deepcopy of the SmartSimEntity object"""
        self.launch_settings = launch_settings
        """Deepcopy of the LaunchSettings object"""
        self._name = name if name else entity.name
        """Name of the Job"""
        check_name(self._name)

    @property
    def name(self) -> str:
        """Return the name of the Job.

        :return: the name of the Job
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Set the name of the Job.

        :param name: the name of the Job
        """
        check_name(name)
        logger.debug(f'Overwriting the Job name from "{self._name}" to "{name}"')
        self._name = name

    @property
    def entity(self) -> SmartSimEntity:
        """Return the attached entity.

        :return: the attached SmartSimEntity
        """
        return deepcopy(self._entity)

    @entity.setter
    def entity(self, value: SmartSimEntity) -> None:
        """Set the Job entity.

        :param value: the SmartSimEntity
        :raises Type Error: if entity is not SmartSimEntity
        """
        from smartsim.entity.entity import SmartSimEntity

        if not isinstance(value, SmartSimEntity):
            raise TypeError("entity argument was not of type SmartSimEntity")

        self._entity = deepcopy(value)

    @property
    def launch_settings(self) -> LaunchSettings:
        """Return the attached LaunchSettings.

        :return: the attached LaunchSettings
        """
        return deepcopy(self._launch_settings)

    @launch_settings.setter
    def launch_settings(self, value: LaunchSettings) -> None:
        """Set the Jobs LaunchSettings.

        :param value: the LaunchSettings
        :raises Type Error: if launch_settings is not a LaunchSettings
        """
        if not isinstance(value, LaunchSettings):
            raise TypeError("launch_settings argument was not of type LaunchSettings")

        self._launch_settings = deepcopy(value)

    def get_launch_steps(self) -> LaunchCommands:
        """Return the launch steps corresponding to the
        internal data.

        :returns: The Jobs launch steps
        """
        # TODO: return JobWarehouseRunner.run(self)
        raise NotImplementedError

    def __str__(self) -> str:  # pragma: no cover
        string = f"SmartSim Entity: {self.entity}\n"
        string += f"Launch Settings: {self.launch_settings}"
        return string
