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

import pathlib
import subprocess
import typing as t
from collections import namedtuple
from datetime import datetime

from ...entity import entity
from ...launchable import Job
from ...log import get_logger
from ..commands import Command, CommandList
from .operations.operations import (
    ConfigureOperation,
    CopyOperation,
    FileSysOperationSet,
    GenerationContext,
    SymlinkOperation,
)

logger = get_logger(__name__)
logger.propagate = False


@t.runtime_checkable
class _GenerableProtocol(t.Protocol):
    """Protocol to ensure that an entity supports both file operations
    and parameters."""

    files: FileSysOperationSet
    # TODO change when file_parameters taken off Application during Ensemble refactor ticket
    file_parameters: t.Mapping[str, str]


Job_Path = namedtuple("Job_Path", ["run_path", "out_path", "err_path"])
"""Namedtuple that stores a Job's run directory, output file path, and
error file path."""


class Generator:
    """The Generator class creates the directory structure for a SmartSim Job by building
    and executing file operation commands.
    """

    run_directory = "run"
    """The name of the directory storing run-related files."""
    log_directory = "log"
    """The name of the directory storing log-related files."""

    def __init__(self, root: pathlib.Path) -> None:
        """Initialize a Generator object

        The Generator class is responsible for constructing a Job's directory, performing
        the following tasks:

        - Creating the run and log directories
        - Generating the output and error files
        - Building the parameter settings file
        - Managing symlinking, copying, and configuration of attached files

        :param root: The base path for job-related files and directories
        """
        self.root = root
        """The root directory under which all generated files and directories will be placed."""

    def _build_job_base_path(self, job: Job, job_index: int) -> pathlib.Path:
        """Build and return a Job's base directory. The path is created by combining the
        root directory with the Job type (derived from the class name),
        the name attribute of the Job, and an index to differentiate between multiple
        Job runs.

        :param job: Job object
        :param job_index: Job index
        :returns: The built file path for the Job
        """
        job_type = f"{job.__class__.__name__.lower()}s"
        job_path = self.root / f"{job_type}/{job.name}-{job_index}"
        return pathlib.Path(job_path)

    def _build_job_run_path(self, job: Job, job_index: int) -> pathlib.Path:
        """Build and return a Job's run directory. The path is formed by combining
        the base directory with the `run_directory` class-level constant, which specifies
        the name of the Job's run folder.

        :param job: Job object
        :param job_index: Job index
        :returns: The built file path for the Job run folder
        """
        path = self._build_job_base_path(job, job_index) / self.run_directory
        return pathlib.Path(path)

    def _build_job_log_path(self, job: Job, job_index: int) -> pathlib.Path:
        """Build and return a Job's log directory. The path is formed by combining
        the base directory with the `log_directory` class-level constant, which specifies
        the name of the Job's log folder.

        :param job: Job object
        :param job_index: Job index
        :returns: The built file path for the Job run folder
        """
        path = self._build_job_base_path(job, job_index) / self.log_directory
        return pathlib.Path(path)

    @staticmethod
    def _build_log_file_path(log_path: pathlib.Path) -> pathlib.Path:
        """Build and return a parameters file summarizing the parameters
        used for the generation of the entity.

        :param log_path: Path to log directory
        :returns: The built file path an entities params file
        """
        return pathlib.Path(log_path) / "smartsim_params.txt"

    @staticmethod
    def _build_out_file_path(log_path: pathlib.Path, job_name: str) -> pathlib.Path:
        """Build and return the path to the output file. The path is created by combining
        the Job's log directory with the job name and appending the `.out` extension.

        :param log_path: Path to log directory
        :param job_name: Name of the Job
        :returns: Path to the output file
        """
        out_file_path = log_path / f"{job_name}.out"
        return out_file_path

    @staticmethod
    def _build_err_file_path(log_path: pathlib.Path, job_name: str) -> pathlib.Path:
        """Build and return the path to the error file. The path is created by combining
        the Job's log directory with the job name and appending the `.err` extension.

        :param log_path: Path to log directory
        :param job_name: Name of the Job
        :returns: Path to the error file
        """
        err_file_path = log_path / f"{job_name}.err"
        return err_file_path

    def generate_job(self, job: Job, job_index: int) -> Job_Path:
        """Build and return the Job's run directory, output file, and error file.

        This method creates the Job's run and log directories, generates the
        `smartsim_params.txt` file to log parameters used for the Job, and sets
        up the output and error files for Job execution information. If files are
        attached to the Job's entity, it builds file operation commands and executes
        them.

        :param job: Job object
        :param job_index: Job index
        :return: Job's run directory, error file and out file.
        """

        job_path = self._build_job_run_path(job, job_index)
        log_path = self._build_job_log_path(job, job_index)

        out_file = self._build_out_file_path(log_path, job.entity.name)
        err_file = self._build_err_file_path(log_path, job.entity.name)

        cmd_list = self._build_commands(job.entity, job_path, log_path)

        self._execute_commands(cmd_list)

        with open(
            self._build_log_file_path(log_path), mode="w", encoding="utf-8"
        ) as log_file:
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_file.write(f"Generation start date and time: {dt_string}\n")

        return Job_Path(job_path, out_file, err_file)

    @classmethod
    def _build_commands(
        cls,
        entity: entity.SmartSimEntity,
        job_path: pathlib.Path,
        log_path: pathlib.Path,
    ) -> CommandList:
        """Build file operation commands for a Job's entity.

        This method constructs commands for copying, symlinking, and writing tagged files
        associated with the Job's entity. This method builds the constructs the commands to
        generate the Job's run and log directory. It aggregates these commands into a CommandList
        to return.

        :param job: Job object
        :param job_path: The file path for the Job run folder
        :param log_path: The file path for the Job log folder
        :return: A CommandList containing the file operation commands
        """
        context = GenerationContext(job_path)
        cmd_list = CommandList()

        cls._append_mkdir_commands(cmd_list, job_path, log_path)

        if isinstance(entity, _GenerableProtocol):
            cls._append_file_operations(cmd_list, entity, context)

        return cmd_list

    @classmethod
    def _append_mkdir_commands(
        cls, cmd_list: CommandList, job_path: pathlib.Path, log_path: pathlib.Path
    ) -> None:
        """Append file operation Commands (mkdir) for a Job's run and log directory.

        :param cmd_list: A CommandList object containing the commands to be executed
        :param job_path: The file path for the Job run folder
        :param log_path: The file path for the Job log folder
        """
        cmd_list.append(cls._mkdir_file(job_path))
        cmd_list.append(cls._mkdir_file(log_path))

    @classmethod
    def _append_file_operations(
        cls,
        cmd_list: CommandList,
        entity: _GenerableProtocol,
        context: GenerationContext,
    ) -> None:
        """Append file operation Commands (copy, symlink, configure) for all
        files attached to the entity.

        :param cmd_list: A CommandList object containing the commands to be executed
        :param entity: The Job's attached entity
        :param context: A GenerationContext object that holds the Job's run directory
        """
        copy_ret = cls._copy_files(entity.files.copy_operations, context)
        cmd_list.extend(copy_ret)

        symlink_ret = cls._symlink_files(entity.files.symlink_operations, context)
        cmd_list.extend(symlink_ret)

        configure_ret = cls._configure_files(entity.files.configure_operations, context)
        cmd_list.extend(configure_ret)

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
    def _mkdir_file(file_path: pathlib.Path) -> Command:
        """Build a Command to create the directory along with any
        necessary parent directories.

        :param file_path: The directory path to be created
        :return: A Command object to execute the directory creation
        """
        cmd = Command(["mkdir", "-p", str(file_path)])
        return cmd

    @staticmethod
    def _copy_files(
        files: list[CopyOperation], context: GenerationContext
    ) -> CommandList:
        """Build commands to copy files/directories from specified source paths
        to an optional destination in the run directory.

        :param files: A list of CopyOperation objects
        :param context: A GenerationContext object that holds the Job's run directory
        :return: A CommandList containing the copy commands
        """
        return CommandList([file.format(context) for file in files])

    @staticmethod
    def _symlink_files(
        files: list[SymlinkOperation], context: GenerationContext
    ) -> CommandList:
        """Build commands to symlink files/directories from specified source paths
        to an optional destination in the run directory.

        :param files: A list of SymlinkOperation objects
        :param context: A GenerationContext object that holds the Job's run directory
        :return: A CommandList containing the symlink commands
        """
        return CommandList([file.format(context) for file in files])

    @staticmethod
    def _configure_files(
        files: list[ConfigureOperation],
        context: GenerationContext,
    ) -> CommandList:
        """Build commands to configure files/directories from specified source paths
        to an optional destination in the run directory.

        :param files: A list of ConfigurationOperation objects
        :param context: A GenerationContext object that holds the Job's run directory
        :return: A CommandList containing the configuration commands
        """
        return CommandList([file.format(context) for file in files])
