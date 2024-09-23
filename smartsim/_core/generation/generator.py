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

import base64
import os
import pathlib
import pickle
import subprocess
import sys
import typing as t
from collections import namedtuple
from datetime import datetime

from ...entity.files import EntityFiles
from ...launchable import Job
from ...log import get_logger
from ..commands import Command, CommandList

logger = get_logger(__name__)
logger.propagate = False


@t.runtime_checkable
class _GenerableProtocol(t.Protocol):
    """Ensures functions using job.entity continue if attrs file and params are supported."""

    files: t.Union[EntityFiles, None]
    file_parameters: t.Mapping[str, str]


Job_Path = namedtuple("Job_Path", ["run_path", "out_path", "err_path"])
"""Paths related to the Jobs execution."""


class Generator:
    """The primary responsibility of the Generator class is to create the directory structure
    for a SmartSim Job and to build and execute file operation commands."""

    run = "run"
    """The name of the directory where run-related files are stored."""
    log = "log"
    """The name of the directory where log files are stored."""

    def __init__(self, root: pathlib.Path) -> None:
        """Initialize a Generator object

        The Generator class constructs a Jobs directory structure, including:

        - The run and log directories
        - Output and error files
        - The "smartsim_params.txt" settings file

        Additionally, it manages symlinking, copying, and configuring files associated
        with a Jobs entity.

        :param root: The Jobs base path
        """
        self.root = root
        """The root path under which to generate files"""

    def _build_job_base_path(self, job: Job, job_index: int) -> pathlib.Path:
        """Build and return a Jobs base directory. The path is created by combining the
        root directory with the Job type (derived from the class name),
        the name attribute of the Job, and an index to differentiate between multiple
        Job runs.

        :param job: The job object
        :param job_index: The index of the Job
        :returns: The built file path for the Job
        """
        job_type = f"{job.__class__.__name__.lower()}s"
        job_path = self.root / f"{job_type}/{job.name}-{job_index}"
        return pathlib.Path(job_path)

    def _build_job_run_path(self, job: Job, job_index: int) -> pathlib.Path:
        """Build and return a Jobs run directory. The path is formed by combining
        the base directory with the `run` class-level variable, where run specifies
        the name of the job's run folder.

        :param job: The job object
        :param job_index: The index of the Job
        :returns: The built file path for the Job run folder
        """
        path = self._build_job_base_path(job, job_index) / self.run
        path.mkdir(exist_ok=False, parents=True)
        return pathlib.Path(path)

    def _build_job_log_path(self, job: Job, job_index: int) -> pathlib.Path:
        """Build and return a Jobs log directory. The path is formed by combining
        the base directory with the `log` class-level variable, where log specifies
        the name of the job's log folder.

        :param job: The job object
        :param job_index: The index of the Job
        :returns: The built file path for the Job run folder
        """
        path = self._build_job_base_path(job, job_index) / self.log
        path.mkdir(exist_ok=False, parents=True)
        return pathlib.Path(path)

    @staticmethod
    def _build_log_file_path(log_path: pathlib.Path) -> pathlib.Path:
        """Build and return an entities file summarizing the parameters
        used for the generation of the entity.

        :param log_path: Path to log directory
        :returns: The built file path an entities params file
        """
        return pathlib.Path(log_path) / "smartsim_params.txt"

    @staticmethod
    def _build_out_file_path(log_path: pathlib.Path, job_name: str) -> pathlib.Path:
        """Build and return the path to the output file. The path is created by combining
        the Jobs log directory with the job name and appending the `.out` extension.

        :param log_path: Path to log directory
        :param job_name: Name of the Job
        :returns: Path to the output file
        """
        out_file_path = log_path / f"{job_name}.out"
        return out_file_path

    @staticmethod
    def _build_err_file_path(log_path: pathlib.Path, job_name: str) -> pathlib.Path:
        """Build and return the path to the error file. The path is created by combining
        the Jobs log directory with the job name and appending the `.err` extension.

        :param log_path: Path to log directory
        :param job_name: Name of the Job
        :returns: Path to the error file
        """
        err_file_path = log_path / f"{job_name}.err"
        return err_file_path

    def generate_job(self, job: Job, job_index: int) -> Job_Path:
        """Build and return the Job's run directory, error file and out file.

        This method creates the Job's run and log directories, generates the
        `smartsim_params.txt` file to log parameters used for the Job, and sets
        up the output and error files for Job execution information. If files are
        attached to the Job's entity, it builds file operation commands and executes
        them.

        :param job: The job object
        :param job_index: The index of the Job
        :return: Job's run directory, error file and out file.
        """

        job_path = self._build_job_run_path(job, job_index)
        log_path = self._build_job_log_path(job, job_index)

        with open(
            self._build_log_file_path(log_path), mode="w", encoding="utf-8"
        ) as log_file:
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f"Generation start date and time: {dt_string}\n")

        out_file = self._build_out_file_path(log_path, job.entity.name)
        err_file = self._build_err_file_path(log_path, job.entity.name)

        cmd_list = self._build_commands(job, job_path)
        if cmd_list:
            self._execute_commands(cmd_list)

        return Job_Path(job_path, out_file, err_file)

    @classmethod
    def _build_commands(
        cls, job: Job, job_path: pathlib.Path
    ) -> t.Optional[CommandList]:
        """Build file operation commands for all files attached to a Job's entity.

        This method constructs commands for copying, symlinking, and writing tagged files
        associated with the Job's entity. It aggregates these commands into a CommandList
        to return.

        :param job: The job object
        :param job_path: The file path for the Job run folder
        :return: A CommandList containing the file operation commands, or None if the entity
            does not support file operations.
        """
        if isinstance(job.entity, _GenerableProtocol):
            cmd_list = CommandList()
            helpers = [
                cls._copy_files,
                cls._symlink_files,
                lambda files, path: cls._write_tagged_files(
                    files, job.entity.file_parameters, path
                ),
            ]

            for method in helpers:
                return_cmd_list = method(job.entity.files, job_path)
                if return_cmd_list:
                    cmd_list.commands.extend(return_cmd_list.commands)

            return cmd_list
        return None

    @classmethod
    def _execute_commands(cls, cmd_list: CommandList) -> None:
        """Execute a list of commands using subprocess.

        This helper function iterates through each command in the provided CommandList
        and executes them using the subprocess module.

        :param cmd_list: A CommandList object containing the commands to be executed
        """
        for cmd in cmd_list:
            subprocess.run(cmd.command)

    @staticmethod
    def _copy_files(
        files: t.Union[EntityFiles, None], dest: pathlib.Path
    ) -> t.Optional[CommandList]:
        """Build command to copy files/directories from specified paths to a destination directory.

        This method creates commands to copy files/directories from the source paths provided in the
        `files` parameter to the specified destination directory. If the source is a directory,
        it copies the directory while allowing existing directories to remain intact.

        :param files: An EntityFiles object containing the paths to copy, or None.
        :param dest: The destination path to the Job's run directory.
        :return: A CommandList containing the copy commands, or None if no files are provided.
        """
        if files is None:
            return None
        cmd_list = CommandList()
        for src in files.copy:
            cmd = Command(
                [
                    sys.executable,
                    "-m",
                    "smartsim._core.entrypoints.file_operations",
                    "copy",
                    src,
                ]
            )
            destination = str(dest)
            if os.path.isdir(src):
                base_source_name = os.path.basename(src)
                destination = os.path.join(dest, base_source_name)
                cmd.append(str(destination))
                cmd.append("--dirs_exist_ok")
            else:
                cmd.append(str(dest))
            cmd_list.commands.append(cmd)
        return cmd_list

    @staticmethod
    def _symlink_files(
        files: t.Union[EntityFiles, None], dest: pathlib.Path
    ) -> t.Optional[CommandList]:
        """Build command to symlink files/directories from specified paths to a destination directory.

        This method creates commands to symlink files/directories from the source paths provided in the
        `files` parameter to the specified destination directory. If the source is a directory,
        it copies the directory while allowing existing directories to remain intact.

        :param files: An EntityFiles object containing the paths to symlink, or None.
        :param dest: The destination path to the Job's run directory.
        :return: A CommandList containing the symlink commands, or None if no files are provided.
        """
        if files is None:
            return None
        cmd_list = CommandList()
        for src in files.link:
            # Normalize the path to remove trailing slashes
            normalized_path = os.path.normpath(src)
            # Get the parent directory (last folder)
            parent_dir = os.path.basename(normalized_path)
            new_dest = os.path.join(str(dest), parent_dir)
            cmd = Command(
                [
                    sys.executable,
                    "-m",
                    "smartsim._core.entrypoints.file_operations",
                    "symlink",
                    src,
                    new_dest,
                ]
            )
            cmd_list.append(cmd)
        return cmd_list

    @staticmethod
    def _write_tagged_files(
        files: t.Union[EntityFiles, None],
        params: t.Mapping[str, str],
        dest: pathlib.Path,
    ) -> t.Optional[CommandList]:
        """Build command to configure files/directories from specified paths to a destination directory.

        This method processes tagged files by reading their configurations,
        serializing the provided parameters, and generating commands to
        write these configurations to the destination directory.

        :param files: An EntityFiles object containing the paths to configure, or None.
        :param params: A dictionary of params
        :param dest: The destination path to the Job's run directory.
        :return: A CommandList containing the configuration commands, or None if no files are provided.
        """
        if files is None:
            return None
        cmd_list = CommandList()
        if files.tagged:
            tag_delimiter = ";"
            pickled_dict = pickle.dumps(params)
            encoded_dict = base64.b64encode(pickled_dict).decode("ascii")
            for path in files.tagged:
                cmd = Command(
                    [
                        sys.executable,
                        "-m",
                        "smartsim._core.entrypoints.file_operations",
                        "configure",
                        path,
                        str(dest),
                        tag_delimiter,
                        encoded_dict,
                    ]
                )
                cmd_list.commands.append(cmd)
        return cmd_list

        # TODO address in ticket 723
        # self._log_params(entity, files_to_params)

    # TODO to be refactored in ticket 723
    # def _log_params(
    #     self, entity: Application, files_to_params: t.Dict[str, t.Dict[str, str]]
    # ) -> None:
    #     """Log which files were modified during generation

    #     and what values were set to the parameters

    #     :param entity: the application being generated
    #     :param files_to_params: a dict connecting each file to its parameter settings
    #     """
    #     used_params: t.Dict[str, str] = {}
    #     file_to_tables: t.Dict[str, str] = {}
    #     for file, params in files_to_params.items():
    #         used_params.update(params)
    #         table = tabulate(params.items(), headers=["Name", "Value"])
    #         file_to_tables[relpath(file, self.gen_path)] = table

    #     if used_params:
    #         used_params_str = ", ".join(
    #             [f"{name}={value}" for name, value in used_params.items()]
    #         )
    #         logger.log(
    #             level=self.log_level,
    #             msg=f"Configured application {entity.name} with params {used_params_str}",
    #         )
    #         file_table = tabulate(
    #             file_to_tables.items(),
    #             headers=["File name", "Parameters"],
    #         )
    #         log_entry = f"Application name: {entity.name}\n{file_table}\n\n"
    #         with open(self.log_file, mode="a", encoding="utf-8") as logfile:
    #             logfile.write(log_entry)
    #         with open(
    #             join(entity.path, "smartsim_params.txt"), mode="w", encoding="utf-8"
    #         ) as local_logfile:
    #             local_logfile.write(log_entry)

    #     else:
    #         logger.log(
    #             level=self.log_level,
    #             msg=f"Configured application {entity.name} with no parameters",
    #         )
