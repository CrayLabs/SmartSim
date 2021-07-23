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
from itertools import product

from ...error import SSConfigError
from ...utils import get_logger
from .step import Step

logger = get_logger(__name__)


class BsubBatchStep(Step):
    def __init__(self, name, cwd, batch_settings):
        """Initialize a LSF bsub step

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
        return [self.batch_settings.batch_cmd, script]

    def add_to_batch(self, step):
        """Add a job step to this batch

        :param step: a job step instance e.g. SrunStep
        :type step: Step
        """
        launch_cmd = step.get_launch_cmd()
        self.step_cmds.append(launch_cmd)
        logger.debug(f"Added step command to batch for {step.name}")

    def _write_script(self):
        """Write the batch script

        :return: batch script path after writing
        :rtype: str
        """
        batch_script = self.get_step_file(ending=".sh")
        output, error = self.get_output_files()

        self.batch_settings._format_alloc_flags()

        opts = self.batch_settings.format_batch_args()

        # if self.batch_settings.expert_mode:
        #     for opt in opts:
        #         if opt in batch_settings.easy_options:
        #             del batch_settings[arg]

        with open(batch_script, "w") as f:
            f.write("#!/bin/bash\n\n")
            if self.batch_settings.walltime:
                f.write(f"#BSUB -W {self.batch_settings.walltime}\n")
            if self.batch_settings.project:
                f.write(f"#BSUB -P {self.batch_settings.project}\n")
            f.write(f"#BSUB -o {output}\n")
            f.write(f"#BSUB -e {error}\n")
            f.write(f"#BSUB -J {self.name}\n")

            # add additional bsub options
            for opt in opts:
                f.write(f"#BSUB {opt}\n")

            for i, cmd in enumerate(self.step_cmds):
                f.write("\n")
                f.write(f"{' '.join((cmd))} &\n")
                if i == len(self.step_cmds) - 1:
                    f.write("\n")
                    f.write("wait\n")
        return batch_script


class JsrunStep(Step):
    def __init__(self, name, cwd, run_settings):
        """Initialize a LSF jsrun job step

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
        jsrun = self.run_settings.run_command

        output, error = self.get_output_files()

        jsrun_cmd = [
            jsrun,
            "--chdir",
            self.cwd,
            "--stdio_stdout",
            output,
            "--stdio_stderr",
            error,
        ]

        if self.run_settings.env_vars:
            env_var_str = self.run_settings.format_env_vars()
            jsrun_cmd += [env_var_str]

        jsrun_cmd += self.run_settings.format_run_args()
        jsrun_cmd += self._build_exe()

        return jsrun_cmd

    def _set_alloc(self):
        """Set the id of the allocation

        :raises SSConfigError: allocation not listed or found
        """
        if "LSB_JOBID" in os.environ:
            self.alloc = os.environ["LSB_JOBID"]
            logger.debug(
                f"Running on LSF allocation {self.alloc} gleaned from user environment"
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
        exe = self.run_settings.exe
        args = self.run_settings.exe_args
        if self.run_settings.mpmd:
            mpmd_file = self.get_step_file(ending=".mpmd")
            cmd = self._make_mpmd(exe, args)
            mp_cmd = ["--erf_input", mpmd_file]
            return mp_cmd
        else:
            cmd = exe + args
            return cmd

    def _make_mpmd(self, executable, exe_args):
        """Build LSF multi-prog (MPMD) executable

        Launch multiple programs on separate cpus on multiple hosts using the
        LSF --erf_input feature. The number of apps must be divisible
        by the number of hosts in run_settings
        """

        mpmd_file = self.get_step_file(ending=".mpmd")
        launch_args = list(product(executable, exe_args))

        if self.run_settings.hosts:
            hosts = self.run_settings.hosts
        else:
            hosts = ["*"]

        tasks_per_host = len(launch_args) // len(hosts)

        with open(mpmd_file, "w+") as f:
            # if host == "*" or int(host) == 0:
            f.write("launch_distribution : packed\n")

            # First we list apps
            app_id = 0
            for exe, args in launch_args:
                e_args = " ".join(args)
                f.write(f"app {app_id} : " + " ".join((exe, e_args, "\n")))
                app_id += 1

            f.write("\n")

            # Then we list resources
            ntasks = self.run_settings.run_args.get("tasks_per_rs", 1)

            smts_per_task = self.run_settings.smts_per_task

            app_id = 0

            old_host = None
            for taskid, launch_arg in enumerate(launch_args):
                exe, args = launch_arg
                host = hosts[taskid // tasks_per_host]
                if host != old_host:
                    assigned_smts = 0
                    assigned_gpus = 0
                old_host = host
                f.write(f"{ntasks} : ")
                f.write("{")
                f.write(f"host: {host}; cpu: ")
                smt_sets = []
                for _ in range(ntasks):
                    smt_set = "{" + f"{assigned_smts}:{smts_per_task}" + "}"
                    smt_sets.append(smt_set)
                    assigned_smts += smts_per_task
                f.write(",".join(smt_sets))
                if "gpu_per_rs" in self.run_settings.run_args.keys():
                    ngpus = self.run_settings.run_args["gpu_per_rs"]
                    f.write("; gpu: ")
                    if ngpus > 1:
                        gpu_set = "{" + f"{assigned_gpus}-{assigned_gpus+ngpus-1}" + "}"
                    else:
                        gpu_set = "{" + f"{assigned_gpus}" + "}"
                    assigned_gpus += ngpus
                    f.write(gpu_set)
                

                f.write("}" + f" : app {app_id} \n")
                app_id += 1

        import sys

        return mpmd_file
