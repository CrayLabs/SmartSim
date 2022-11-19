# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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
from shlex import split as sh_split

from ....error import AllocationError
from ....log import get_logger
from .step import Step

logger = get_logger(__name__)

class _BaseMPIStep(Step):
    def __init__(self, name, cwd, run_settings):
        """Initialize a job step conforming to the MPI standard

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param run_settings: run settings for entity
        :type run_settings: RunSettings
        """

        super().__init__(name, cwd)


        self.run_settings = run_settings

        self.alloc = None
        if not self.run_settings.in_batch:
            self._set_alloc()

    _supported_launchers = [
        "PBS",
        "COBALT",
        "SLURM",
        "LSB"
    ]

    @property
    def _run_command(self):
        return self.run_settings._run_command

    def get_launch_cmd(self):
        """Get the command to launch this step

        :return: launch command
        :rtype: list[str]
        """
        mpi_cmd = [self._run_command, "--wdir", self.cwd]
        # add env vars to mpi command
        mpi_cmd.extend(self.run_settings.format_env_vars())

        # add mpi settings to command
        mpi_cmd.extend(self.run_settings.format_run_args())

        if self.run_settings.colocated_db_settings:
            # disable cpu binding as the entrypoint will set that
            # for the application and database process now
            # mpi_cmd.extend(["--cpu-bind", "none"])

            # Replace the command with the entrypoint wrapper script
            bash = shutil.which("bash")
            launch_script_path = self.get_colocated_launch_script()
            mpi_cmd += [bash, launch_script_path]

        mpi_cmd += self._build_exe()

        # if its in a batch, redirect stdout to
        # file in the cwd.
        if self.run_settings.in_batch:
            output = self.get_step_file(ending=".out")
            mpi_cmd += [">", output]
        return mpi_cmd

    def _set_alloc(self):
        """Set the id of the allocation

        :raises AllocationError: allocation not listed or found
        """

        environment_keys = os.environ.keys()
        for launcher in self._supported_launchers:
            jobid_field = f'{launcher.upper()}_JOBID'
            if jobid_field in environment_keys:
                self.alloc = os.environ[jobid_field]
                logger.debug(
                    f"Running on allocation {self.alloc} from {jobid_field}"
                )
                return

        # If this function did not return above, no allocations were found
        raise AllocationError(
            "No allocation specified or found and not running in batch"
        )

    def _build_exe(self):
        """Build the executable for this step

        :return: executable list
        :rtype: list[str]
        """
        if self.run_settings.mpmd:
            return self._make_mpmd()
        else:
            exe = self.run_settings.exe
            args = self.run_settings.exe_args
            return exe + args

    def _make_mpmd(self):
        """Build mpiexec (MPMD) executable"""
        exe = self.run_settings.exe
        args = self.run_settings.exe_args
        cmd = exe + args
        for mpmd in self.run_settings.mpmd:
            cmd += [" : "]
            cmd += mpmd.format_run_args()
            cmd += mpmd.format_env_vars()
            cmd += mpmd.exe
            cmd += mpmd.exe_args

        cmd = sh_split(" ".join(cmd))
        return cmd

class MpiexecStep(_BaseMPIStep):
    def __init__(self, name, cwd, run_settings):
        """Initialize an mpiexec job step

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param run_settings: run settings for entity
        :type run_settings: RunSettings
        :param default_run_command: The default command to launch an MPI
                                    application
        :type default_run_command: str, optional
        """

        super().__init__(name, cwd, run_settings)


class MpirunStep(_BaseMPIStep):
    def __init__(self, name, cwd, run_settings):
        """Initialize an mpirun job step

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param run_settings: run settings for entity
        :type run_settings: RunSettings
        :param default_run_command: The default command to launch an MPI
                                    application
        :type default_run_command: str, optional
        """

        super().__init__(name, cwd, run_settings)

class OrterunStep(_BaseMPIStep):
    def __init__(self, name, cwd, run_settings):
        """Initialize an orterun job step

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param run_settings: run settings for entity
        :type run_settings: RunSettings
        :param default_run_command: The default command to launch an MPI
                                    application
        :type default_run_command: str, optional
        """

        super().__init__(name, cwd, run_settings)