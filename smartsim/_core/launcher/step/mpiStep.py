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
from shlex import split as sh_split
import typing as t

from ....error import AllocationError, SmartSimError
from ....log import get_logger
from .step import Step
from ....settings import MpirunSettings, MpiexecSettings, OrterunSettings
from ....settings.base import RunSettings

logger = get_logger(__name__)


class _BaseMPIStep(Step):
    def __init__(self, name: str, cwd: str, run_settings: RunSettings) -> None:
        """Initialize a job step conforming to the MPI standard

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param run_settings: run settings for entity
        :type run_settings: RunSettings
        """

        super().__init__(name, cwd, run_settings)

        self.alloc: t.Optional[str] = None
        if not run_settings.in_batch:
            self._set_alloc()
        self.run_settings = run_settings

    _supported_launchers = ["PBS", "COBALT", "SLURM", "LSB"]

    def get_launch_cmd(self) -> t.List[str]:
        """Get the command to launch this step

        :return: launch command
        :rtype: list[str]
        """
        run_cmd = self.run_settings.run_command
        if not run_cmd:
            raise SmartSimError("No run command specified")

        mpi_cmd = [run_cmd, "--wdir", self.cwd]
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
            if not bash:
                raise RuntimeError("Could not find bash in PATH")
            launch_script_path = self.get_colocated_launch_script()
            mpi_cmd += [bash, launch_script_path]

        mpi_cmd += self._build_exe()

        # if its in a batch, redirect stdout to
        # file in the cwd.
        if self.run_settings.in_batch:
            output = self.get_step_file(ending=".out")
            mpi_cmd += [">", output]
        return mpi_cmd

    def _set_alloc(self) -> None:
        """Set the id of the allocation

        :raises AllocationError: allocation not listed or found
        """

        environment_keys = os.environ.keys()
        for launcher in self._supported_launchers:
            jobid_field = f"{launcher.upper()}_JOBID"
            if jobid_field in environment_keys:
                self.alloc = os.environ[jobid_field]
                logger.debug(f"Running on allocation {self.alloc} from {jobid_field}")
                return

        # If this function did not return above, no allocations were found
        raise AllocationError(
            "No allocation specified or found and not running in batch"
        )

    def _get_mpmd(self) -> t.List[RunSettings]:
        """Temporary convenience function to return a typed list
        of attached RunSettings"""
        if hasattr(self.run_settings, "mpmd") and self.run_settings.mpmd:
            rs_mpmd: t.List[RunSettings] = self.run_settings.mpmd
            return rs_mpmd
        return []

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
        """Build mpiexec (MPMD) executable"""
        exe = self.run_settings.exe
        args = self.run_settings._exe_args  # pylint: disable=protected-access
        cmd = exe + args

        for mpmd in self._get_mpmd():
            cmd += [" : "]
            cmd += mpmd.format_run_args()
            cmd += mpmd.format_env_vars()
            cmd += mpmd.exe
            cmd += mpmd._exe_args  # pylint: disable=protected-access

        cmd = sh_split(" ".join(cmd))
        return cmd


class MpiexecStep(_BaseMPIStep):
    def __init__(self, name: str, cwd: str, run_settings: MpiexecSettings) -> None:
        """Initialize an mpiexec job step

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param run_settings: run settings for entity
        :type run_settings: MpiexecSettings
        :param default_run_command: The default command to launch an MPI
                                    application
        :type default_run_command: str, optional
        """

        super().__init__(name, cwd, run_settings)


class MpirunStep(_BaseMPIStep):
    def __init__(self, name: str, cwd: str, run_settings: MpirunSettings) -> None:
        """Initialize an mpirun job step

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param run_settings: run settings for entity
        :type run_settings: MpirunSettings
        :param default_run_command: The default command to launch an MPI
                                    application
        :type default_run_command: str, optional
        """

        super().__init__(name, cwd, run_settings)


class OrterunStep(_BaseMPIStep):
    def __init__(self, name: str, cwd: str, run_settings: OrterunSettings) -> None:
        """Initialize an orterun job step

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param run_settings: run settings for entity
        :type run_settings: OrterunSettings
        :param default_run_command: The default command to launch an MPI
                                    application
        :type default_run_command: str, optional
        """

        super().__init__(name, cwd, run_settings)
