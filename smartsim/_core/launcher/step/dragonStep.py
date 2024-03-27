# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import shutil
import typing as t

from ....log import get_logger
from ....settings import DragonRunSettings, Singularity
from .step import Step

logger = get_logger(__name__)


class DragonStep(Step):
    def __init__(self, name: str, cwd: str, run_settings: DragonRunSettings) -> None:
        """Initialize a srun job step

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param run_settings: run settings for entity
        :type run_settings: SrunSettings
        """
        super().__init__(name, cwd, run_settings)
        self.managed = True

    @property
    def run_settings(self) -> DragonRunSettings:
        return t.cast(DragonRunSettings, self.step_settings)

    def get_launch_cmd(self) -> t.List[str]:
        """Get stringified version of request
         needed to launch this step

        :return: launch command
        :rtype: list[str]
        """
        run_settings = self.run_settings
        exe_cmd = []

        if run_settings.colocated_db_settings:
            # Replace the command with the entrypoint wrapper script
            bash = shutil.which("bash")
            if not bash:
                raise RuntimeError("Could not find bash in PATH")
            launch_script_path = self.get_colocated_launch_script()
            exe_cmd += [bash, launch_script_path]

        if isinstance(run_settings.container, Singularity):
            # pylint: disable-next=protected-access
            exe_cmd += run_settings.container._container_cmds(self.cwd)

        exe_cmd += run_settings.exe

        exe_args = self._get_exe_args_list(run_settings)

        exe_cmd_and_args = exe_cmd + exe_args

        return exe_cmd_and_args

    @staticmethod
    def _get_exe_args_list(run_setting: DragonRunSettings) -> t.List[str]:
        """Convenience function to encapsulate checking the
        runsettings.exe_args type to always return a list
        """
        exe_args = run_setting.exe_args
        args: t.List[str] = exe_args if isinstance(exe_args, list) else [exe_args]
        return args
