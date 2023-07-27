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

from __future__ import annotations

import shutil
import subprocess
import typing as t

from ..error import LauncherError, SSUnsupportedError
from ..log import get_logger
from .base import RunSettings

logger = get_logger(__name__)


class _BaseMPISettings(RunSettings):
    """Base class for all common arguments of MPI-standard run commands"""

    def __init__(
        self,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.List[str]]] = None,
        run_command: str = "mpiexec",
        run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        fail_if_missing_exec: bool = True,
        **kwargs: t.Any,
    ) -> None:
        """Settings to format run job with an MPI-standard binary

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
        :param fail_if_missing_exec: Throw an exception of the MPI command
                                     is missing. Otherwise, throw a warning
        :type fail_if_missing_exec: bool, optional
        """
        super().__init__(
            exe,
            exe_args,
            run_command=run_command,
            run_args=run_args,
            env_vars=env_vars,
            **kwargs,
        )
        self.mpmd: t.List[RunSettings] = []

        if not shutil.which(self._run_command):
            msg = (
                f"Cannot find {self._run_command}. Try passing the "
                "full path via run_command."
            )
            if fail_if_missing_exec:
                raise LauncherError(msg)
            logger.warning(msg)

    reserved_run_args = {"wd", "wdir"}

    def make_mpmd(self, settings: RunSettings) -> None:
        """Make a mpmd workload by combining two ``mpirun`` commands

        This connects the two settings to be executed with a single
        Model instance

        :param settings: MpirunSettings instance
        :type settings: MpirunSettings
        """
        if self.colocated_db_settings:
            raise SSUnsupportedError(
                "Colocated models cannot be run as a mpmd workload"
            )
        self.mpmd.append(settings)

    def set_task_map(self, task_mapping: str) -> None:
        """Set ``mpirun`` task mapping

        this sets ``--map-by <mapping>``

        For examples, see the man page for ``mpirun``

        :param task_mapping: task mapping
        :type task_mapping: str
        """
        self.run_args["map-by"] = task_mapping

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of tasks for this job

        This sets ``--cpus-per-proc`` for MPI compliant implementations

        note: this option has been deprecated in openMPI 4.0+
        and will soon be replaced.

        :param cpus_per_task: number of tasks
        :type cpus_per_task: int
        """
        self.run_args["cpus-per-proc"] = int(cpus_per_task)

    def set_cpu_binding_type(self, bind_type: str) -> None:
        """Specifies the cores to which MPI processes are bound

        This sets ``--bind-to`` for MPI compliant implementations

        :param bind_type: binding type
        :type bind_type: str
        """
        self.run_args["bind-to"] = bind_type

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks per node

        :param tasks_per_node: number of tasks to launch per node
        :type tasks_per_node: int
        """
        self.run_args["npernode"] = int(tasks_per_node)

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        This sets ``-n`` for MPI compliant implementations

        :param tasks: number of tasks
        :type tasks: int
        """
        self.run_args["n"] = int(tasks)

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
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
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self.run_args["host"] = ",".join(host_list)

    def set_hostlist_from_file(self, file_path: str) -> None:
        """Use the contents of a file to set the hostlist

        This sets ``--hostfile``

        :param file_path: Path to the hostlist file
        :type file_path: str
        """
        self.run_args["hostfile"] = file_path

    def set_verbose_launch(self, verbose: bool) -> None:
        """Set the job to run in verbose mode

        This sets ``--verbose``

        :param verbose: Whether the job should be run verbosely
        :type verbose: bool
        """
        if verbose:
            self.run_args["verbose"] = None
        else:
            self.run_args.pop("verbose", None)

    def set_quiet_launch(self, quiet: bool) -> None:
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        :type quiet: bool
        """
        if quiet:
            self.run_args["quiet"] = None
        else:
            self.run_args.pop("quiet", None)

    def set_broadcast(self, dest_path: t.Optional[str] = None) -> None:
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

    def set_walltime(self, walltime: str) -> None:
        """Set the maximum number of seconds that a job will run

        This sets ``--timeout``

        :param walltime: number like string of seconds that a job will run in secs
        :type walltime: str
        """
        self.run_args["timeout"] = walltime

    def format_run_args(self) -> t.List[str]:
        """Return a list of MPI-standard formatted run arguments

        :return: list of MPI-standard arguments for these settings
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

    def format_env_vars(self) -> t.List[str]:
        """Format the environment variables for mpirun

        :return: list of env vars
        :rtype: list[str]
        """
        formatted = []
        env_string = "-x"

        if self.env_vars:
            for name, value in self.env_vars.items():
                if value:
                    formatted += [env_string, "=".join((name, str(value)))]
                else:
                    formatted += [env_string, name]
        return formatted


class MpirunSettings(_BaseMPISettings):
    def __init__(
        self,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.List[str]]] = None,
        run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ) -> None:
        """Settings to run job with ``mpirun`` command (MPI-standard)

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
        :type run_args: dict[str, t.Union[int, str, float, None]], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(exe, exe_args, "mpirun", run_args, env_vars, **kwargs)


class MpiexecSettings(_BaseMPISettings):
    def __init__(
        self,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.List[str]]] = None,
        run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ) -> None:
        """Settings to run job with ``mpiexec`` command (MPI-standard)

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
        :type run_args: dict[str, t.Union[int, str, float, None]], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(exe, exe_args, "mpiexec", run_args, env_vars, **kwargs)

        completed_process = subprocess.run(
            [self._run_command, "--help"], capture_output=True, check=False
        )
        help_statement = completed_process.stdout.decode()
        if "mpiexec.slurm" in help_statement:
            raise SSUnsupportedError(
                "Slurm's wrapper for mpiexec is unsupported. Use slurmSettings instead"
            )


class OrterunSettings(_BaseMPISettings):
    def __init__(
        self,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.List[str]]] = None,
        run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ) -> None:
        """Settings to run job with ``orterun`` command (MPI-standard)

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
        :type run_args: dict[str, t.Union[int, str, float, None]], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(exe, exe_args, "orterun", run_args, env_vars, **kwargs)
