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

from __future__ import annotations

import typing as t

from ..log import get_logger
from .base import RunSettings

logger = get_logger(__name__)


class DragonRunSettings(RunSettings):
    def __init__(
        self,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.List[str]]] = None,
        run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        alloc: t.Optional[str] = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialize run parameters for a Dragon process

        ``DragonRunSettings`` should only be used on systems where Dragon
        is available.

        If an allocation is specified, the instance receiving these run
        parameters will launch on that allocation.

        :param exe: executable to run
        :param exe_args: executable arguments, defaults to None
        :param run_args: srun arguments without dashes, defaults to None
        :param env_vars: environment variables for job, defaults to None
        :param alloc: allocation ID if running on existing alloc, defaults to None
        """
        super().__init__(
            exe,
            exe_args,
            run_command="",
            run_args=run_args,
            env_vars=env_vars,
            **kwargs,
        )
        self.alloc = alloc
        self.mpmd: t.List[RunSettings] = []

    def set_nodes(self, nodes: int) -> None:
        """Set the number of nodes

        :param nodes: number of nodes to run with
        """
        self.run_args["nodes"] = nodes

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self.run_args["nodelist"] = ",".join(host_list)

    def set_hostlist_from_file(self, file_path: str) -> None:
        """Use the contents of a file to set the node list

        :param file_path: Path to the hostlist file
        """
        self.run_args["nodefile"] = file_path

    def set_excluded_hosts(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify a list of hosts to exclude for launching this job

        :param host_list: hosts to exclude
        :raises TypeError:
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self.run_args["exclude"] = ",".join(host_list)

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of cpus to use per task

        :param num_cpus: number of cpus to use per task
        """
        self.run_args["cpus-per-task"] = cpus_per_task

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        :param tasks: number of tasks
        """
        self.run_args["ntasks"] = tasks

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks for this job

        :param tasks_per_node: number of tasks per node
        """
        self.run_args["tasks-per-node"] = tasks_per_node

    def set_memory_per_node(self, memory_per_node: int) -> None:
        """Specify the real memory required per node

        :param memory_per_node: Amount of memory per node in megabytes
        """
        self.run_args["mem"] = f"{memory_per_node}M"

    def set_verbose_launch(self, verbose: bool) -> None:
        """Set the job to run in verbose mode

        This sets ``--verbose``

        :param verbose: Whether the job should be run verbosely
        """
        if verbose:
            self.run_args["verbose"] = None
        else:
            self.run_args.pop("verbose", None)

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        format = "HH:MM:SS"

        :param walltime: wall time
        """
        self.run_args["time"] = str(walltime)
