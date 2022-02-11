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

from ....error import AllocationError
from ....log import get_logger
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

        with open(batch_script, "w") as f:
            f.write("#!/bin/bash\n\n")
            if self.batch_settings.walltime:
                f.write(f"#BSUB -W {self.batch_settings.walltime}\n")
            if self.batch_settings.project:
                f.write(f"#BSUB -P {self.batch_settings.project}\n")
            f.write(f"#BSUB -J {self.name}\n")
            f.write(f"#BSUB -o {output}\n")
            f.write(f"#BSUB -e {error}\n")

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
        self.managed = True
        if not self.run_settings.in_batch:
            self._set_alloc()

    def get_output_files(self):
        """Return two paths to error and output files based on cwd"""
        output = self.get_step_file(ending=".out")
        error = self.get_step_file(ending=".err")

        # The individual_suffix (containing %t and similar placeholders) is
        # appended to the output (and error) file name, but just before the ending.
        # So if the collective output file name would be "entity_name.out", adding
        # a typical suffix "_%t", will result in "entity_name_%t.out" passed to
        # --stdio_stdout (similarly for error). This in turn, will be processed
        # by jsrun, replacing each occurrence of "%t" with the task number and
        # writing output to "entity_name_0.out", "entity_name_1.out"...
        if self.run_settings.individual_suffix:
            partitioned_output = output.rpartition(".")
            output_prefix = partitioned_output[0] + self.run_settings.individual_suffix
            output_suffix = partitioned_output[-1]
            output = ".".join([output_prefix, output_suffix])
            partitioned_error = error.rpartition(".")
            error_prefix = partitioned_error[0] + self.run_settings.individual_suffix
            error_suffix = partitioned_error[-1]
            error = ".".join([error_prefix, error_suffix])

        return output, error

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
            env_var_str_list = self.run_settings.format_env_vars()
            jsrun_cmd.extend(env_var_str_list)

        jsrun_cmd.extend(self.run_settings.format_run_args())

        if self.run_settings.colocated_db_settings:
            # disable cpu binding as the entrypoint will set that
            # for the application and database process now
            jsrun_cmd.extend(["--bind", "none"])

            # Replace the command with the entrypoint wrapper script
            bash = shutil.which("bash")
            launch_script_path = self.get_colocated_launch_script()
            jsrun_cmd.extend([bash, launch_script_path])

        jsrun_cmd.extend(self._build_exe())

        return jsrun_cmd

    def _set_alloc(self):
        """Set the id of the allocation

        :raises AllocationError: allocation not listed or found
        """
        if "LSB_JOBID" in os.environ:
            self.alloc = os.environ["LSB_JOBID"]
            logger.debug(
                f"Running on LSF allocation {self.alloc} gleaned from user environment"
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
        exe = self.run_settings.exe
        args = self.run_settings.exe_args
        if self.run_settings.mpmd:
            erf_file = self.get_step_file(ending=".mpmd")
            cmd = self._make_mpmd()
            mp_cmd = ["--erf_input", erf_file]
            return mp_cmd
        else:
            cmd = exe + args
            return cmd

    def _make_mpmd(self):
        """Build LSF's Explicit Resource File"""
        erf_file = self.get_step_file(ending=".mpmd")

        # Find launch_distribution command
        preamble_lines = self.run_settings.mpmd_preamble_lines.copy()
        distr_line = None
        for line in self.run_settings.mpmd_preamble_lines:
            if line.lstrip(" ").startswith("launch_distribution"):
                distr_line = line
                preamble_lines.remove(line)
        if not distr_line:
            for jrs in self.run_settings.mpmd:
                if "launch_distribution" in jrs.run_args.keys():
                    distr_line = (
                        "launch_distribution : " + jrs.run_args["launch_distribution"]
                    )
                elif "d" in jrs.run_args.keys():
                    distr_line = "launch_distribution : " + jrs.run_args["d"]
                if distr_line:
                    break
        if not distr_line:
            distr_line = "launch_distribution : packed"

        with open(erf_file, "w+") as f:
            f.write(distr_line + "\n")
            for line in preamble_lines:
                f.write(line + "\n")
            f.write("\n")

            # First we list the apps
            for app_id, jrs in enumerate(self.run_settings.mpmd):
                f.write(f"app {app_id} : " + " ".join(jrs.exe + jrs.exe_args) + "\n")
            f.write("\n")

            # Then we list the resources
            for app_id, jrs in enumerate(self.run_settings.mpmd):
                rs_line = ""
                if "rank" in jrs.erf_sets.keys():
                    rs_line += "rank: " + jrs.erf_sets["rank"] + ": "
                elif "rank_count" in jrs.erf_sets.keys():
                    rs_line += jrs.erf_sets["rank_count"] + ": "
                else:
                    rs_line += "1 : "

                rs_line += "{ "
                if "host" in jrs.erf_sets.keys():
                    rs_line += "host: " + jrs.erf_sets["host"] + "; "
                else:
                    rs_line += "host: *;"

                if "cpu" in jrs.erf_sets.keys():
                    rs_line += "cpu: " + jrs.erf_sets["cpu"]
                else:
                    rs_line += "cpu: * "

                if "gpu" in jrs.erf_sets.keys():
                    rs_line += "; gpu: " + jrs.erf_sets["gpu"]

                if "memory" in jrs.erf_sets.keys():
                    rs_line += "; memory: " + jrs.erf_sets["memory"]

                rs_line += "}: app " + str(app_id) + "\n"

                f.write(rs_line)

        with open(erf_file) as f:
            f.flush()
            os.fsync(f)
