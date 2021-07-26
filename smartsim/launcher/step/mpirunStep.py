# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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

from ...error import SSConfigError
from ...utils import get_logger
from .step import Step

logger = get_logger(__name__)


class MpirunStep(Step):
    def __init__(self, name, cwd, run_settings):
        """Initialize a OpenMPI mpirun job step

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

    def get_launch_cmd(self):
        """Get the command to launch this step

        :return: launch command
        :rtype: list[str]
        """
        mpirun = self.run_settings.run_command
        mpirun_cmd = [mpirun, "-wdir", self.cwd]
        mpirun_cmd += self.run_settings.format_env_vars()
        mpirun_cmd += self._build_exe()

        # if its in a batch, redirect stdout to
        # file in the cwd.
        if self.run_settings.in_batch:
            output = self.get_step_file(ending=".out")
            mpirun_cmd += [">", output]
        return mpirun_cmd

    def _set_alloc(self):
        """Set the id of the allocation

        :raises SSConfigError: allocation not listed or found
        """
        if "PBS_JOBID" in os.environ:  # cov-pbs
            self.alloc = os.environ["PBS_JOBID"]
            logger.debug(
                f"Running on PBS allocation {self.alloc} gleaned from user environment"
            )
        elif "COBALT_JOBID" in os.environ:  # cov-cobalt
            self.alloc = os.environ["COBALT_JOBID"]
            logger.debug(
                f"Running on Cobalt allocation {self.alloc} gleaned from user environment"
            )
        elif "SLURM_JOBID" in os.environ:  # cov-slurm
            self.alloc = os.environ["SLURM_JOBID"]
            logger.debug(
                f"Running on Slurm allocation {self.alloc} gleaned from user environment"
            )
        elif "LSB_JOBID" in os.environ:  # cov-lsf
            self.alloc = os.environ["LSB_JOBID"]
            logger.debug(
                f"Running on Slurm allocation {self.alloc} gleaned from user environment"
            )
        else:
            raise SSConfigError(
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
            cmd = self.run_settings.format_run_args()
            cmd += self.run_settings.exe
            cmd += self.run_settings.exe_args
            return cmd

    def _make_mpmd(self):
        """Build mpirun (MPMD) executable"""
        cmd = self.run_settings.format_run_args()
        cmd += self.run_settings.exe
        cmd += self.run_settings.exe_args
        for mpmd in self.run_settings.mpmd:
            cmd += [" : "]
            cmd += mpmd.format_run_args()
            cmd += mpmd.format_env_vars()
            cmd += mpmd.exe
            cmd += mpmd.exe_args
        return cmd
