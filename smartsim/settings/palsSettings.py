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

import typing as t

from ..log import get_logger
from .mpiSettings import _BaseMPISettings

logger = get_logger(__name__)


class PalsMpiexecSettings(_BaseMPISettings):
    """Settings to run job with ``mpiexec`` under the HPE Cray
    Parallel Application Launch Service (PALS)

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

    def __init__(
        self,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.List[str]]] = None,
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
        :type run_args: dict[str, t.Union[int, str, float, None]], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        :param fail_if_missing_exec: Throw an exception of the MPI command
                                     is missing. Otherwise, throw a warning
        :type fail_if_missing_exec: bool, optional
        """
        super().__init__(
            exe,
            exe_args,
            run_command="mpiexec",
            run_args=run_args,
            env_vars=env_vars,
            fail_if_missing_exec=fail_if_missing_exec,
            **kwargs,
        )

    def set_task_map(self, task_mapping: str) -> None:
        """Set ``mpirun`` task mapping

        this sets ``--map-by <mapping>``

        For examples, see the man page for ``mpirun``

        :param task_mapping: task mapping
        :type task_mapping: str
        """
        logger.warning("set_task_map not supported under PALS")

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of tasks for this job

        This sets ``--cpus-per-proc`` for MPI compliant implementations

        note: this option has been deprecated in openMPI 4.0+
        and will soon be replaced.

        :param cpus_per_task: number of tasks
        :type cpus_per_task: int
        """
        logger.warning("set_cpus_per_task not supported under PALS")

    def set_cpu_binding_type(self, bind_type: str) -> None:
        """Specifies the cores to which MPI processes are bound

        This sets ``--bind-to`` for MPI compliant implementations

        :param bind_type: binding type
        :type bind_type: str
        """
        self.run_args["cpu-bind"] = bind_type

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks

        :param tasks: number of total tasks to launch
        :type tasks: int
        """
        self.run_args["np"] = int(tasks)

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks per node

        :param tasks_per_node: number of tasks to launch per node
        :type tasks_per_node: int
        """
        self.run_args["ppn"] = int(tasks_per_node)

    def set_quiet_launch(self, quiet: bool) -> None:
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        :type quiet: bool
        """

        logger.warning("set_quiet_launch not supported under PALS")

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
        self.run_args["transfer"] = None

    def set_walltime(self, walltime: str) -> None:
        """Set the maximum number of seconds that a job will run

        :param walltime: number like string of seconds that a job will run in secs
        :type walltime: str
        """
        logger.warning("set_walltime not supported under PALS")

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

        export_vars = []
        if self.env_vars:
            for name, value in self.env_vars.items():
                if value:
                    formatted += ["--env", "=".join((name, str(value)))]
                else:
                    export_vars.append(name)

        if export_vars:
            formatted += ["--envlist", ",".join(export_vars)]

        return formatted
