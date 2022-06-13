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

import subprocess
import re

from ..error import SSUnsupportedError
from ..log import get_logger
from .base import RunSettings

logger = get_logger(__name__)


class _OpenMPISettings(RunSettings):
    """Base class for all common arguments of OpenMPI run commands"""

    def __init__(
        self, exe, exe_args=None, run_command="", run_args=None, env_vars=None, **kwargs
    ):
        """Settings to format run job with an OpenMPI binary

        Note that environment variables can be passed with a None
        value to signify that they should be exported from the current
        environment

        Any arguments passed in the ``run_args`` dict will be converted
        command line arguments and prefixed with ``--``. Values of
        None can be provided for arguments that do not have values.

        :param exe: executable
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: str | list[str], optional
        :param run_args: arguments for run command, defaults to None
        :type run_args: dict[str, str], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(
            exe,
            exe_args,
            run_command=run_command,
            run_args=run_args,
            env_vars=env_vars,
            **kwargs,
        )
        self.mpmd = []

    reserved_run_args = {"wd", "wdir"}

    def make_mpmd(self, mpirun_settings):
        """Make a mpmd workload by combining two ``mpirun`` commands

        This connects the two settings to be executed with a single
        Model instance

        :param mpirun_settings: MpirunSettings instance
        :type mpirun_settings: MpirunSettings
        """
        if self.colocated_db_settings:
            raise SSUnsupportedError(
                "Colocated models cannot be run as a mpmd workload"
            )
        self.mpmd.append(mpirun_settings)

    def set_task_map(self, task_mapping):
        """Set ``mpirun`` task mapping

        this sets ``--map-by <mapping>``

        For examples, see the man page for ``mpirun``

        :param task_mapping: task mapping
        :type task_mapping: str
        """
        self.run_args["map-by"] = str(task_mapping)

    def set_cpus_per_task(self, cpus_per_task):
        """Set the number of tasks for this job

        This sets ``--cpus-per-proc``

        note: this option has been deprecated in openMPI 4.0+
        and will soon be replaced.

        :param cpus_per_task: number of tasks
        :type cpus_per_task: int
        """
        self.run_args["cpus-per-proc"] = int(cpus_per_task)

    def set_tasks_per_node(self, tasks_per_node):
        """Set the number of tasks per node

        :param tasks_per_node: number of tasks to launch per node
        :type tasks_per_node: int
        """
        self.run_args["npernode"] = int(tasks_per_node)

    def set_tasks(self, tasks):
        """Set the number of tasks for this job

        This sets ``--n``

        :param tasks: number of tasks
        :type tasks: int
        """
        self.run_args["n"] = int(tasks)

    def set_hostlist(self, host_list):
        """Set the hostlist for the ``mpirun`` command

        This sets ``--host``

        :param host_list: list of host names
        :type host_list: str | list[str]
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all([isinstance(host, str) for host in host_list]):
            raise TypeError("host_list argument must be list of strings")
        self.run_args["host"] = ",".join(host_list)

    def set_hostlist_from_file(self, file_path):
        """Use the contents of a file to set the hostlist

        This sets ``--hostfile``

        :param file_path: Path to the hostlist file
        :type file_path: str
        """
        self.run_args["hostfile"] = str(file_path)

    def set_verbose_launch(self, verbose):
        """Set the job to run in verbose mode

        This sets ``--verbose``

        :param verbose: Whether the job should be run verbosely
        :type verbose: bool
        """
        if verbose:
            self.run_args["verbose"] = None
        else:
            self.run_args.pop("verbose", None)

    def set_quiet_launch(self, quiet):
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        :type quiet: bool
        """
        if quiet:
            self.run_args["quiet"] = None
        else:
            self.run_args.pop("quiet", None)

    def set_broadcast(self, dest_path=None):
        """Copy the specified executable(s) to remote machines

        This sets ``--preload-binary``

        :param dest_path: Destination path (Ignored)
        :type dest_path: str | None
        """
        if dest_path is not None and isinstance(dest_path, str):
            logger.warning(
                (
                    f"{type(self)} cannot set a destination path during broadcast. "
                    "Using session directory instead"
                )
            )
        self.run_args["preload-binary"] = None

    def set_walltime(self, walltime):
        """Set the maximum number of seconds that a job will run

        This sets ``--timeout``

        :param walltime: number like string of seconds that a job will run in secs
        :type walltime: str
        """
        self.run_args["timeout"] = str(walltime)

    def format_run_args(self):
        """Return a list of OpenMPI formatted run arguments

        :return: list of OpenMPI arguments for these settings
        :rtype: list[str]
        """
        # args launcher uses
        args = []
        restricted = ["wdir", "wd"]

        for opt, value in self.run_args.items():
            if opt not in restricted:
                prefix = "--"
                if not value:
                    args += [prefix + opt]
                else:
                    args += [prefix + opt, str(value)]
        return args

    def format_env_vars(self):
        """Format the environment variables for mpirun

        :return: list of env vars
        :rtype: list[str]
        """
        formatted = []

        if self.env_vars:
            for name, value in self.env_vars.items():
                if value:
                    formatted += ["-x", "=".join((name, str(value)))]
                else:
                    formatted += ["-x", name]
        return formatted


class MpirunSettings(_OpenMPISettings):
    def __init__(self, exe, exe_args=None, run_args=None, env_vars=None, **kwargs):
        """Settings to run job with ``mpirun`` command (OpenMPI)

        Note that environment variables can be passed with a None
        value to signify that they should be exported from the current
        environment

        Any arguments passed in the ``run_args`` dict will be converted
        into ``mpirun`` arguments and prefixed with ``--``. Values of
        None can be provided for arguments that do not have values.

        :param exe: executable
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: str | list[str], optional
        :param run_args: arguments for run command, defaults to None
        :type run_args: dict[str, str], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(exe, exe_args, "mpirun", run_args, env_vars, **kwargs)

        completed_process = subprocess.run(
            [self.run_command, "-V"], capture_output=True
        )  # type: subprocess.CompletedProcess
        version_statement = completed_process.stdout.decode()

        if not re.match(r"mpirun\s\(Open MPI\)\s4.\d+.\d+", version_statement):
            logger.warning("Non-OpenMPI implementation of `mpirun` detected")


class MpiexecSettings(_OpenMPISettings):
    def __init__(self, exe, exe_args=None, run_args=None, env_vars=None, **kwargs):
        """Settings to run job with ``mpiexec`` command (OpenMPI)

        Note that environment variables can be passed with a None
        value to signify that they should be exported from the current
        environment

        Any arguments passed in the ``run_args`` dict will be converted
        into ``mpiexec`` arguments and prefixed with ``--``. Values of
        None can be provided for arguments that do not have values.

        :param exe: executable
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: str | list[str], optional
        :param run_args: arguments for run command, defaults to None
        :type run_args: dict[str, str], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(exe, exe_args, "mpiexec", run_args, env_vars, **kwargs)

        completed_process = subprocess.run(
            [self.run_command, "-V"], capture_output=True
        )  # type: subprocess.CompletedProcess
        version_statement = completed_process.stdout.decode()

        if not re.match(r"mpiexec\s\(OpenRTE\)\s4.\d+.\d+", version_statement):
            logger.warning("Non-OpenMPI implementation of `mpiexec` detected")


class OrterunSettings(_OpenMPISettings):
    def __init__(self, exe, exe_args=None, run_args=None, env_vars=None, **kwargs):
        """Settings to run job with ``orterun`` command (OpenMPI)

        Note that environment variables can be passed with a None
        value to signify that they should be exported from the current
        environment

        Any arguments passed in the ``run_args`` dict will be converted
        into ``orterun`` arguments and prefixed with ``--``. Values of
        None can be provided for arguments that do not have values.

        :param exe: executable
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: str | list[str], optional
        :param run_args: arguments for run command, defaults to None
        :type run_args: dict[str, str], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(exe, exe_args, "orterun", run_args, env_vars, **kwargs)

        completed_process = subprocess.run(
            [self.run_command, "-V"], capture_output=True
        )  # type: subprocess.CompletedProcess
        version_statement = completed_process.stdout.decode()

        if not re.match(r"orterun\s\(OpenRTE\)\s4.\d+.\d+", version_statement):
            logger.warning("Non-OpenMPI implementation of `orterun` detected")
