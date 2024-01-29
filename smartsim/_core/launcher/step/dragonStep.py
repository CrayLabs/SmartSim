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

import os
import shlex
import shutil
import typing as t

from ....log import get_logger
from ....settings import DragonRunSettings, Singularity
from ...schemas import DragonRunRequest
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
        self.alloc: t.Optional[str] = None
        self.managed = True
        self.run_settings = run_settings

    def get_launch_request(self) -> DragonRunRequest:
        """Get the command to launch this step

        :return: launch request
        :rtype: DragonRunRequest
        """

        output, error = self.get_output_files()

        rs = self.run_settings

        if "nodes" in rs._run_args:
            nodes = t.cast(int, rs._run_args["nodes"])
        else:
            nodes = 1

        exe_cmd = []

        if self.run_settings.colocated_db_settings:
            # Replace the command with the entrypoint wrapper script
            bash = shutil.which("bash")
            if not bash:
                raise RuntimeError("Could not find bash in PATH")
            launch_script_path = self.get_colocated_launch_script()
            exe_cmd += [bash, launch_script_path]

        if isinstance(self.run_settings.container, Singularity):
            # pylint: disable-next=protected-access
            exe_cmd += self.run_settings.container._container_cmds(self.cwd)

        exe_cmd += rs.exe

        exe_args = self._get_exe_args_list(rs)

        run_request = DragonRunRequest(
            exe=exe_cmd,
            exe_args=exe_args,
            path=self.cwd,
            nodes=nodes,
            output_file=output,
            error_file=error,
            env=rs.env_vars,
            name=self.name,
        )

        return run_request

    @staticmethod
    def _get_exe_args_list(run_setting: DragonRunSettings) -> t.List[str]:
        """Convenience function to encapsulate checking the
        runsettings.exe_args type to always return a list
        """
        exe_args = run_setting.exe_args
        args: t.List[str] = exe_args if isinstance(exe_args, list) else [exe_args]
        return args