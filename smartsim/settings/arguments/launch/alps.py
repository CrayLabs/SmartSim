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

from smartsim._core.arguments.shell import ShellLaunchArguments
from smartsim._core.dispatch import dispatch
from smartsim._core.shell.shell_launcher import ShellLauncher, make_shell_format_fn
from smartsim.log import get_logger

from ...common import set_check_input
from ...launch_command import LauncherType

logger = get_logger(__name__)
_as_aprun_command = make_shell_format_fn(run_command="aprun")


@dispatch(with_format=_as_aprun_command, to_launcher=ShellLauncher)
class AprunLaunchArguments(ShellLaunchArguments):
    def _reserved_launch_args(self) -> set[str]:
        """Return reserved launch arguments.

        :returns: The set of reserved launcher arguments
        """
        return {"wdir"}

    def launcher_str(self) -> str:
        """Get the string representation of the launcher

        :returns: The string representation of the launcher
        """
        return LauncherType.Alps.value

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of cpus to use per task

        This sets ``--cpus-per-pe``

        :param cpus_per_task: number of cpus to use per task
        """
        self.set("cpus-per-pe", str(cpus_per_task))

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        This sets ``--pes``

        :param tasks: number of tasks
        """
        self.set("pes", str(tasks))

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks for this job

        This sets ``--pes-per-node``

        :param tasks_per_node: number of tasks per node
        """
        self.set("pes-per-node", str(tasks_per_node))

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        This sets ``--node-list``

        :param host_list: hosts to launch on
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self.set("node-list", ",".join(host_list))

    def set_hostlist_from_file(self, file_path: str) -> None:
        """Use the contents of a file to set the node list

        This sets ``--node-list-file``

        :param file_path: Path to the hostlist file
        """
        self.set("node-list-file", file_path)

    def set_excluded_hosts(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify a list of hosts to exclude for launching this job

        This sets ``--exclude-node-list``

        :param host_list: hosts to exclude
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self.set("exclude-node-list", ",".join(host_list))

    def set_cpu_bindings(self, bindings: t.Union[int, t.List[int]]) -> None:
        """Specifies the cores to which MPI processes are bound

        This sets ``--cpu-binding``

        :param bindings: List of cpu numbers
        """
        if isinstance(bindings, int):
            bindings = [bindings]
        self.set("cpu-binding", ",".join(str(num) for num in bindings))

    def set_memory_per_node(self, memory_per_node: int) -> None:
        """Specify the real memory required per node

        This sets ``--memory-per-pe`` in megabytes

        :param memory_per_node: Per PE memory limit in megabytes
        """
        self.set("memory-per-pe", str(memory_per_node))

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        Walltime is given in total number of seconds

        :param walltime: wall time
        """
        self.set("cpu-time-limit", str(walltime))

    def set_verbose_launch(self, verbose: bool) -> None:
        """Set the job to run in verbose mode

        This sets ``--debug`` arg to the highest level

        :param verbose: Whether the job should be run verbosely
        """
        if verbose:
            self.set("debug", "7")
        else:
            self._launch_args.pop("debug", None)

    def set_quiet_launch(self, quiet: bool) -> None:
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        """
        if quiet:
            self._launch_args["quiet"] = None
        else:
            self._launch_args.pop("quiet", None)

    def format_env_vars(self, env_vars: t.Mapping[str, str | None]) -> list[str]:
        """Format the environment variables for aprun

        :return: list of env vars
        """
        formatted = []
        if env_vars:
            for name, value in env_vars.items():
                formatted += ["-e", name + "=" + str(value)]
        return formatted

    def format_launch_args(self) -> t.List[str]:
        """Return a list of ALPS formatted run arguments

        :return: list of ALPS arguments for these settings
        """
        # args launcher uses
        args = []
        for opt, value in self._launch_args.items():
            short_arg = len(opt) == 1
            prefix = "-" if short_arg else "--"
            if not value:
                args += [prefix + opt]
            else:
                if short_arg:
                    args += [prefix + opt, str(value)]
                else:
                    args += ["=".join((prefix + opt, str(value)))]
        return args

    def set(self, key: str, value: str | None) -> None:
        """Set an arbitrary launch argument

        :param key: The launch argument
        :param value: A string representation of the value for the launch
            argument (if applicable), otherwise `None`
        """
        set_check_input(key, value)
        if key in self._reserved_launch_args():
            logger.warning(
                (
                    f"Could not set argument '{key}': "
                    f"it is a reserved argument of '{type(self).__name__}'"
                )
            )
            return
        if key in self._launch_args and key != self._launch_args[key]:
            logger.warning(f"Overwritting argument '{key}' with value '{value}'")
        self._launch_args[key] = value
