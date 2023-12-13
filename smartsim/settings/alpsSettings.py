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
import typing as t

from ..error import SSUnsupportedError
from .base import RunSettings


class AprunSettings(RunSettings):
    def __init__(
        self,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.List[str]]] = None,
        run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ):
        """Settings to run job with ``aprun`` command

        ``AprunSettings`` can be used for both the `pbs` and `cobalt`
        launchers.

        :param exe: executable
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: str | list[str], optional
        :param run_args: arguments for run command, defaults to None
        :type run_args: dict[str, t.Union[int, str, float, None]], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(
            exe,
            exe_args,
            run_command="aprun",
            run_args=run_args,
            env_vars=env_vars,
            **kwargs,
        )
        self.mpmd: t.List[RunSettings] = []

    def make_mpmd(self, settings: RunSettings) -> None:
        """Make job an MPMD job

        This method combines two ``AprunSettings``
        into a single MPMD command joined with ':'

        :param settings: ``AprunSettings`` instance
        :type settings: AprunSettings
        """
        if self.colocated_db_settings:
            raise SSUnsupportedError(
                "Colocated models cannot be run as a mpmd workload"
            )
        if self.container:
            raise SSUnsupportedError(
                "Containerized MPMD workloads are not yet supported."
            )
        self.mpmd.append(settings)

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of cpus to use per task

        This sets ``--cpus-per-pe``

        :param cpus_per_task: number of cpus to use per task
        :type cpus_per_task: int
        """
        self.run_args["cpus-per-pe"] = int(cpus_per_task)

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        This sets ``--pes``

        :param tasks: number of tasks
        :type tasks: int
        """
        self.run_args["pes"] = int(tasks)

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks for this job

        This sets ``--pes-per-node``

        :param tasks_per_node: number of tasks per node
        :type tasks_per_node: int
        """
        self.run_args["pes-per-node"] = int(tasks_per_node)

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :type host_list: str | list[str]
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self.run_args["node-list"] = ",".join(host_list)

    def set_hostlist_from_file(self, file_path: str) -> None:
        """Use the contents of a file to set the node list

        This sets ``--node-list-file``

        :param file_path: Path to the hostlist file
        :type file_path: str
        """
        self.run_args["node-list-file"] = file_path

    def set_excluded_hosts(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify a list of hosts to exclude for launching this job

        :param host_list: hosts to exclude
        :type host_list: str | list[str]
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self.run_args["exclude-node-list"] = ",".join(host_list)

    def set_cpu_bindings(self, bindings: t.Union[int, t.List[int]]) -> None:
        """Specifies the cores to which MPI processes are bound

        This sets ``--cpu-binding``

        :param bindings: List of cpu numbers
        :type bindings: list[int] | int
        """
        if isinstance(bindings, int):
            bindings = [bindings]
        self.run_args["cpu-binding"] = ",".join(str(int(num)) for num in bindings)

    def set_memory_per_node(self, memory_per_node: int) -> None:
        """Specify the real memory required per node

        This sets ``--memory-per-pe`` in megabytes

        :param memory_per_node: Per PE memory limit in megabytes
        :type memory_per_node: int
        """
        self.run_args["memory-per-pe"] = int(memory_per_node)

    def set_verbose_launch(self, verbose: bool) -> None:
        """Set the job to run in verbose mode

        This sets ``--debug`` arg to the highest level

        :param verbose: Whether the job should be run verbosely
        :type verbose: bool
        """
        if verbose:
            self.run_args["debug"] = 7
        else:
            self.run_args.pop("debug", None)

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

    def format_run_args(self) -> t.List[str]:
        """Return a list of ALPS formatted run arguments

        :return: list of ALPS arguments for these settings
        :rtype: list[str]
        """
        # args launcher uses
        args = []
        restricted = ["wdir"]

        for opt, value in self.run_args.items():
            if opt not in restricted:
                short_arg = bool(len(str(opt)) == 1)
                prefix = "-" if short_arg else "--"
                if not value:
                    args += [prefix + opt]
                else:
                    if short_arg:
                        args += [prefix + opt, str(value)]
                    else:
                        args += ["=".join((prefix + opt, str(value)))]
        return args

    def format_env_vars(self) -> t.List[str]:
        """Format the environment variables for aprun

        :return: list of env vars
        :rtype: list[str]
        """
        formatted = []
        if self.env_vars:
            for name, value in self.env_vars.items():
                formatted += ["-e", name + "=" + str(value)]
        return formatted

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        Walltime is given in total number of seconds

        :param walltime: wall time
        :type walltime: str
        """
        self.run_args["cpu-time-limit"] = str(walltime)
