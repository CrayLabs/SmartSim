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

from smartsim._core.commands.launchCommands import LaunchCommands
from smartsim.launchable.basejob import BaseJob
from smartsim.settings import LaunchSettings
from smartsim._core.utils.helpers import create_short_id_str

if t.TYPE_CHECKING:
    from smartsim.entity.entity import SmartSimEntity


class Job(BaseJob):
    """A Job holds a reference to a SmartSimEntity and associated
    LaunchSettings prior to launch.  It is responsible for turning
    the stored entity and launch settings into commands that can be
    executed by a launcher.

    Jobs will hold a deep copy of launch settings.
    """

    def __init__(
        self,
        entity: SmartSimEntity,
        launch_settings: LaunchSettings,
        name: str = "job",
        **kwargs: t.Any,
    ):
        super().__init__()
        self._entity = deepcopy(entity)
        self._launch_settings = deepcopy(launch_settings)
        self._name = deepcopy(name)
        self._ensemble_name = kwargs.get('ensemble_name', None)
        if self._ensemble_name is not None:
            self._ensemble_name += f"-{create_short_id_str()}"
        # TODO: self.warehouse_runner = JobWarehouseRunner

    # TODO do we want the user to be allowed to reset the Job name? Therefore, add setter
    @property
    def name(self) -> str:
        """Retrieves the name of the Job."""
        return deepcopy(self._name)

    @property
    def entity(self) -> SmartSimEntity:
        return deepcopy(self._entity)

    @entity.setter
    def entity(self, value: SmartSimEntity) -> None:
        self._entity = deepcopy(value)

    @property
    def launch_settings(self) -> LaunchSettings:
        return deepcopy(self._launch_settings)

    @launch_settings.setter
    def launch_settings(self, value: LaunchSettings) -> None:
        self._launch_settings = deepcopy(value)

    def get_launch_steps(self) -> LaunchCommands:
        """Return the launch steps corresponding to the
        internal data.
        """
        # TODO: return JobWarehouseRunner.run(self)
        raise NotImplementedError

    def __str__(self) -> str:  # pragma: no cover
        string = f"SmartSim Entity: {self.entity}\n"
        string += f"Launch Settings: {self.launch_settings}"
        return string
