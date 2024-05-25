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

from abc import abstractmethod

from smartsim.entity.entity import SmartSimEntity
from smartsim.launchable.basejob import BaseJob
from smartsim.settings import RunSettings


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
        launch_settings: RunSettings,  # rename to LaunchSettings
    ) -> None:
        super().__init__()
        self.entity = entity
        self.launch_settings = launch_settings
        # self.warehouse_runner = JobWarehouseRunner

    def get_launch_steps(self) -> None:  # -> LaunchCommands:
        """Return the launch steps corresponding to the
        internal data.
        """
        pass
        # return JobWarehouseRunner.run(self)

    def __str__(self) -> str:
        string = f"SmartSim Entity: {self.entity}"
        string += f"Launch Settings: {self.launch_settings}"
        return string
