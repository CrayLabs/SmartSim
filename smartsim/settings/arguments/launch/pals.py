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
_as_pals_command = make_shell_format_fn(run_command="mpiexec")


@dispatch(with_format=_as_pals_command, to_launcher=ShellLauncher)
class PalsMpiexecLaunchArguments(ShellLaunchArguments):
    def launcher_str(self) -> str:
        """Get the string representation of the launcher

        :returns: The string representation of the launcher
        """
        return LauncherType.Pals.value

    def _reserved_launch_args(self) -> set[str]:
        """Return reserved launch arguments.

        :returns: The set of reserved launcher arguments
        """
        return {"wdir", "wd"}

    def set_cpu_binding_type(self, bind_type: str) -> None:
        """Specifies the cores to which MPI processes are bound

        This sets ``--bind-to`` for MPI compliant implementations

        :param bind_type: binding type
        """
        self.set("bind-to", bind_type)

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks

        :param tasks: number of total tasks to launch
        """
        self.set("np", str(tasks))

    def set_executable_broadcast(self, dest_path: str) -> None:
        """Copy the specified executable(s) to remote machines

        This sets ``--transfer``

        :param dest_path: Destination path (Ignored)
        """
        self.set("transfer", dest_path)

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks per node

        This sets ``--ppn``

        :param tasks_per_node: number of tasks to launch per node
        """
        self.set("ppn", str(tasks_per_node))

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Set the hostlist for the PALS ``mpiexec`` command

        This sets ``hosts``

        :param host_list: list of host names
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self.set("hosts", ",".join(host_list))

    def format_env_vars(self, env_vars: t.Mapping[str, str | None]) -> list[str]:
        """Format the environment variables for mpirun

        :return: list of env vars
        """
        formatted = []

        export_vars = []
        if env_vars:
            for name, value in env_vars.items():
                if value:
                    formatted += ["--env", "=".join((name, str(value)))]
                else:
                    export_vars.append(name)

        if export_vars:
            formatted += ["--envlist", ",".join(export_vars)]

        return formatted

    def format_launch_args(self) -> t.List[str]:
        """Return a list of MPI-standard formatted launcher arguments

        :return: list of MPI-standard arguments for these settings
        """
        # args launcher uses
        args = []

        for opt, value in self._launch_args.items():
            prefix = "--"
            if not value:
                args += [prefix + opt]
            else:
                args += [prefix + opt, str(value)]

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
                f"Could not set argument '{key}': "
                f"it is a reserved argument of '{type(self).__name__}'"
            )
            return
        if key in self._launch_args and key != self._launch_args[key]:
            logger.warning(f"Overwritting argument '{key}' with value '{value}'")
        self._launch_args[key] = value
