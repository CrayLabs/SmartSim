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


class SbatchStep(Step):
    def __init__(self, name, cwd, batch_settings):
        """Initialize a Slurm Sbatch step

        :param name: name of the entity to launch
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param batch_settings: batch settings for entity
        :type batch_settings: BatchSettings
        """
        super().__init__(name, cwd)
        self.batch_settings = batch_settings
        self.step_cmds = []
        self.managed = True

    def get_launch_cmd(self):
        """Get the launch command for the batch

        :return: launch command for the batch
        :rtype: list[str]
        """
        script = self._write_script()
        return [self.batch_settings.batch_cmd, "--parsable", script]

    def add_to_batch(self, step):
        """Add a job step to this batch

        :param step: a job step instance e.g. SrunStep
        :type step: Step
        """
        launch_cmd = ["cd", step.cwd, ";"]
        launch_cmd += step.get_launch_cmd()
        self.step_cmds.append(launch_cmd)
        logger.debug(f"Added step command to batch for {step.name}")

    def _write_script(self):
        """Write the batch script

        :return: batch script path after writing
        :rtype: str
        """
        batch_script = self.get_step_file(ending=".sh")
        output, error = self.get_output_files()
        with open(batch_script, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"#SBATCH --output={output}\n")
            f.write(f"#SBATCH --error={error}\n")
            f.write(f"#SBATCH --job-name={self.name}\n")

            # add additional sbatch options
            for opt in self.batch_settings.format_batch_args():
                f.write(f"#SBATCH {opt}\n")

            for cmd in self.batch_settings._preamble:
                f.write(f"{cmd}\n")

            for i, cmd in enumerate(self.step_cmds):
                f.write("\n")
                f.write(f"{' '.join((cmd))} &\n")
                if i == len(self.step_cmds) - 1:
                    f.write("\n")
                    f.write("wait\n")
        return batch_script


class SrunStep(Step):
    def __init__(self, name, cwd, run_settings):
        """Initialize a srun job step

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
        self.managed = True
        if not self.run_settings.in_batch:
            self._set_alloc()

    def get_launch_cmd(self):
        """Get the command to launch this step

        :return: launch command
        :rtype: list[str]
        """
        srun = self.run_settings.run_command
        output, error = self.get_output_files()

        srun_cmd = [srun, "--output", output, "--error", error, "--job-name", self.name]

        if self.alloc:
            srun_cmd += ["--jobid", str(self.alloc)]

        if self.run_settings.env_vars:
            env_var_str, comma_separated_env_vars = self.run_settings.format_env_vars()
            srun_cmd += ["--export", env_var_str]

            if comma_separated_env_vars:
                srun_cmd = ["env"] + comma_separated_env_vars + srun_cmd

        srun_cmd += self.run_settings.format_run_args()

        if self.run_settings.colocated_db_settings:
            # disable cpu binding as the entrypoint will set that
            # for the application and database process now
            srun_cmd.append("--cpu-bind=none")

            # Replace the command with the entrypoint wrapper script
            bash = shutil.which("bash")
            launch_script_path = self.get_colocated_launch_script()
            srun_cmd += [bash, launch_script_path]

        srun_cmd += self._build_exe()
        return srun_cmd

    def _set_alloc(self):
        """Set the id of the allocation

        :raises AllocationError: allocation not listed or found
        """
        if self.run_settings.alloc:
            self.alloc = str(self.run_settings.alloc)
        else:
            if "SLURM_JOB_ID" in os.environ:
                self.alloc = os.environ["SLURM_JOB_ID"]
                logger.debug(
                    f"Running on allocation {self.alloc} gleaned from user environment"
                )
            else:
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
        """Build Slurm multi-prog (MPMD) executable"""
        exe = self.run_settings.exe
        args = self.run_settings.exe_args
        cmd = exe + args
        for mpmd in self.run_settings.mpmd:
            cmd += [" : "]
            cmd += mpmd.format_run_args()
            cmd += ["--job-name", self.name]
            (env_var_str, _) = mpmd.format_env_vars()
            cmd += ["--export", env_var_str]
            cmd += mpmd.exe
            cmd += mpmd.exe_args

        cmd = sh_split(" ".join(cmd))
        return cmd
