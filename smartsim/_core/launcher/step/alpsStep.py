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
import shutil
import typing as t
from shlex import split as sh_split

from ....error import AllocationError
from ....log import get_logger
from .step import Step
from ....settings import AprunSettings, RunSettings, Singularity

logger = get_logger(__name__)


class AprunStep(Step):
    def __init__(self, name: str, cwd: str, run_settings: AprunSettings) -> None:
        """Initialize a ALPS aprun job step

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param run_settings: run settings for entity
        :type run_settings: AprunSettings
        """
        super().__init__(name, cwd, run_settings)
        self.alloc: t.Optional[str] = None
        if not run_settings.in_batch:
            self._set_alloc()
        self.run_settings = run_settings

    def _get_mpmd(self) -> t.List[RunSettings]:
        """Temporary convenience function to return a typed list
        of attached RunSettings"""
        return self.run_settings.mpmd

    def get_launch_cmd(self) -> t.List[str]:
        """Get the command to launch this step

        :return: launch command
        :rtype: list[str]
        """
        aprun = self.run_settings.run_command
        if not aprun:
            logger.warning("aprun not found in PATH")
            raise RuntimeError("Could not find aprun in PATH")

        aprun_cmd = [aprun, "--wdir", self.cwd]

        # add env vars and run settings
        aprun_cmd.extend(self.run_settings.format_env_vars())
        aprun_cmd.extend(self.run_settings.format_run_args())

        if self.run_settings.colocated_db_settings:
            # disable cpu binding as the entrypoint will set that
            # for the application and database process now
            aprun_cmd.extend(["--cc", "none"])

            # Replace the command with the entrypoint wrapper script
            bash = shutil.which("bash")
            if not bash:
                raise RuntimeError("Could not find bash in PATH")
            launch_script_path = self.get_colocated_launch_script()
            aprun_cmd.extend([bash, launch_script_path])

        if isinstance(self.run_settings.container, Singularity):
            # pylint: disable-next=protected-access
            aprun_cmd += self.run_settings.container._container_cmds(self.cwd)

        aprun_cmd += self._build_exe()

        # if its in a batch, redirect stdout to
        # file in the cwd.
        if self.run_settings.in_batch:
            output = self.get_step_file(ending=".out")
            aprun_cmd.extend([">", output])
        return aprun_cmd

    def _set_alloc(self) -> None:
        """Set the id of the allocation

        :raises AllocationError: allocation not listed or found
        """
        if "PBS_JOBID" in os.environ:
            self.alloc = os.environ["PBS_JOBID"]
            logger.debug(
                f"Running on PBS allocation {self.alloc} gleaned from user environment"
            )
        elif "COBALT_JOBID" in os.environ:
            self.alloc = os.environ["COBALT_JOBID"]
            logger.debug(
                f"Running on Cobalt allocation {self.alloc} gleaned "
                "from user environment"
            )
        else:
            raise AllocationError(
                "No allocation specified or found and not running in batch"
            )

    def _build_exe(self) -> t.List[str]:
        """Build the executable for this step

        :return: executable list
        :rtype: list[str]
        """
        if self._get_mpmd():
            return self._make_mpmd()

        exe = self.run_settings.exe
        args = self.run_settings._exe_args  # pylint: disable=protected-access
        return exe + args

    def _make_mpmd(self) -> t.List[str]:
        """Build Aprun (MPMD) executable"""

        exe = self.run_settings.exe
        exe_args = self.run_settings._exe_args  # pylint: disable=protected-access
        cmd = exe + exe_args

        for mpmd in self._get_mpmd():
            cmd += [" : "]
            cmd += mpmd.format_run_args()
            cmd += mpmd.exe
            cmd += mpmd._exe_args  # pylint: disable=protected-access
        cmd = sh_split(" ".join(cmd))
        return cmd
