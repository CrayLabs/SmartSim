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

import json
import os
import shutil
import sys
import typing as t

from ...._core.schemas.dragonRequests import (
    DragonRunPolicy,
    DragonRunRequest,
    request_registry,
)
from ....error.errors import SSUnsupportedError
from ....log import get_logger
from ....settings import (
    DragonRunSettings,
    QsubBatchSettings,
    SbatchSettings,
    Singularity,
)
from .step import Step

logger = get_logger(__name__)


class DragonStep(Step):
    def __init__(self, name: str, cwd: str, run_settings: DragonRunSettings) -> None:
        """Initialize a srun job step

        :param name: name of the entity to be launched
        :param cwd: path to launch dir
        :param run_settings: run settings for entity
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


class DragonBatchStep(Step):
    def __init__(
        self,
        name: str,
        cwd: str,
        batch_settings: t.Union[SbatchSettings, QsubBatchSettings],
    ) -> None:
        """Initialize a Slurm Sbatch step

        :param name: name of the entity to launch
        :param cwd: path to launch dir
        :param batch_settings: batch settings for entity
        """
        super().__init__(name, cwd, batch_settings)
        self.steps: t.List[Step] = []
        self.managed = True
        self.batch_settings = batch_settings
        self._request_file_name = "requests.json"

    def get_launch_cmd(self) -> t.List[str]:
        """Get the launch command for the batch

        :return: launch command for the batch
        """
        if isinstance(self.batch_settings, SbatchSettings):
            script = self._write_sbatch_script()
            return [self.batch_settings.batch_cmd, "--parsable", script]
        if isinstance(self.batch_settings, QsubBatchSettings):
            script = self._write_qsub_script()
            return [self.batch_settings.batch_cmd, script]

        raise SSUnsupportedError(
            "DragonBatchStep only support SbatchSettings and QsubBatchSettings"
        )

    def add_to_batch(self, step: Step) -> None:
        """Add a job step to this batch

        :param step: a job step instance e.g. DragonStep
        """
        self.steps.append(step)
        logger.debug(f"Added step command to batch for {step.name}")

    @staticmethod
    def _dragon_entrypoint_cmd(request_file: str) -> str:
        """Return command needed to run the Dragon entrypoint"""
        cmd = [
            sys.executable,
            "-m",
            "smartsim._core.entrypoints.dragon_client",
            "+submit",
            request_file,
        ]
        return " ".join(cmd)

    def _write_request_file(self) -> str:
        """Write json file with requests to submit to Dragon server"""
        request_file = self.get_step_file(
            ending="json", script_name=self._request_file_name
        )
        requests = []
        for step in self.steps:
            run_settings = t.cast(DragonRunSettings, step.step_settings)
            run_args = run_settings.run_args
            env = run_settings.env_vars
            nodes = int(run_args.get("nodes", None) or 1)
            tasks_per_node = int(run_args.get("tasks-per-node", None) or 1)
            hosts_csv = run_args.get("host-list", None)

            policy = DragonRunPolicy.from_run_args(run_args)

            cmd = step.get_launch_cmd()
            out, err = step.get_output_files()

            request = DragonRunRequest(
                exe=cmd[0],
                exe_args=cmd[1:],
                path=step.cwd,
                name=step.name,
                nodes=nodes,
                tasks_per_node=tasks_per_node,
                env=env,
                current_env=os.environ,
                output_file=out,
                error_file=err,
                policy=policy,
                hostlist=hosts_csv,
            )
            requests.append(request_registry.to_string(request))
        with open(request_file, "w", encoding="utf-8") as script_file:
            script_file.write(json.dumps(requests))

        return request_file

    def _write_sbatch_script(self) -> str:
        """Write the PBS batch script

        :return: batch script path after writing
        """
        batch_script = self.get_step_file(ending=".sh")
        output, error = self.get_output_files()
        request_file = self._write_request_file()
        with open(batch_script, "w", encoding="utf-8") as script_file:
            script_file.write("#!/bin/bash\n\n")
            script_file.write(f"#SBATCH --output={output}\n")
            script_file.write(f"#SBATCH --error={error}\n")
            script_file.write(f"#SBATCH --job-name={self.name}\n")

            # add additional sbatch options
            for opt in self.batch_settings.format_batch_args():
                script_file.write(f"#SBATCH {opt}\n")

            script_file.write(
                f"#SBATCH --export=ALL,SMARTSIM_DRAGON_SERVER_PATH={self.cwd},"
                "PYTHONUNBUFFERED=1\n"
            )

            for cmd in self.batch_settings.preamble:
                script_file.write(f"{cmd}\n")

            script_file.write(
                DragonBatchStep._dragon_entrypoint_cmd(request_file) + "\n"
            )
        return batch_script

    def _write_qsub_script(self) -> str:
        """Write the Slurm batch script

        :return: batch script path after writing
        """
        batch_script = self.get_step_file(ending=".sh")
        output, error = self.get_output_files()
        request_file = self._write_request_file()
        with open(batch_script, "w", encoding="utf-8") as script_file:
            script_file.write("#!/bin/bash\n\n")
            script_file.write(f"#PBS -o {output}\n")
            script_file.write(f"#PBS -e {error}\n")
            script_file.write(f"#PBS -N {self.name}\n")
            script_file.write("#PBS -V \n")

            # add additional sbatch options
            for opt in self.batch_settings.format_batch_args():
                script_file.write(f"#PBS {opt}\n")

            script_file.write(f"#PBS -v SMARTSIM_DRAGON_SERVER_PATH={self.cwd}\n")

            for cmd in self.batch_settings.preamble:
                script_file.write(f"{cmd}\n")

            script_file.write(
                DragonBatchStep._dragon_entrypoint_cmd(request_file) + "\n"
            )

        return batch_script
