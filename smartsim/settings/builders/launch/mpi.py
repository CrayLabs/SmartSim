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

from smartsim.log import get_logger
from smartsim.settings.dispatch import ShellLauncher, dispatch

from ...common import set_check_input
from ...launchCommand import LauncherType
from ..launchArgBuilder import LaunchArgBuilder

if t.TYPE_CHECKING:
    from smartsim.settings.builders.launchArgBuilder import ExecutableLike

logger = get_logger(__name__)


class _BaseMPIArgBuilder(LaunchArgBuilder[t.Sequence[str]]):
    def _reserved_launch_args(self) -> set[str]:
        """Return reserved launch arguments."""
        return {"wd", "wdir"}

    def set_task_map(self, task_mapping: str) -> None:
        """Set ``mpirun`` task mapping

        this sets ``--map-by <mapping>``

        For examples, see the man page for ``mpirun``

        :param task_mapping: task mapping
        """
        self.set("map-by", task_mapping)

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of tasks for this job

        This sets ``--cpus-per-proc`` for MPI compliant implementations

        note: this option has been deprecated in openMPI 4.0+
        and will soon be replaced.

        :param cpus_per_task: number of tasks
        """
        self.set("cpus-per-proc", str(cpus_per_task))

    def set_executable_broadcast(self, dest_path: str) -> None:
        """Copy the specified executable(s) to remote machines

        This sets ``--preload-binary``

        :param dest_path: Destination path (Ignored)
        """
        if dest_path is not None and isinstance(dest_path, str):
            logger.warning(
                (
                    f"{type(self)} cannot set a destination path during broadcast. "
                    "Using session directory instead"
                )
            )
        self.set("preload-binary", dest_path)

    def set_cpu_binding_type(self, bind_type: str) -> None:
        """Specifies the cores to which MPI processes are bound

        This sets ``--bind-to`` for MPI compliant implementations

        :param bind_type: binding type
        """
        self.set("bind-to", bind_type)

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks per node

        :param tasks_per_node: number of tasks to launch per node
        """
        self.set("npernode", str(tasks_per_node))

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        This sets ``-n`` for MPI compliant implementations

        :param tasks: number of tasks
        """
        self.set("n", str(tasks))

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Set the hostlist for the ``mpirun`` command

        This sets ``--host``

        :param host_list: list of host names
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self.set("host", ",".join(host_list))

    def set_hostlist_from_file(self, file_path: str) -> None:
        """Use the contents of a file to set the hostlist

        This sets ``--hostfile``

        :param file_path: Path to the hostlist file
        """
        self.set("hostfile", file_path)

    def set_verbose_launch(self, verbose: bool) -> None:
        """Set the job to run in verbose mode

        This sets ``--verbose``

        :param verbose: Whether the job should be run verbosely
        """
        if verbose:
            self.set("verbose", None)
        else:
            self._launch_args.pop("verbose", None)

    def set_walltime(self, walltime: str) -> None:
        """Set the maximum number of seconds that a job will run

        This sets ``--timeout``

        :param walltime: number like string of seconds that a job will run in secs
        """
        self.set("timeout", walltime)

    def set_quiet_launch(self, quiet: bool) -> None:
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        """
        if quiet:
            self.set("quiet", None)
        else:
            self._launch_args.pop("quiet", None)

    def format_env_vars(
        self, env_vars: t.Optional[t.Dict[str, t.Optional[str]]]
    ) -> t.Union[t.List[str], None]:
        """Format the environment variables for mpirun

        :return: list of env vars
        """
        formatted = []
        env_string = "-x"

        if env_vars:
            for name, value in env_vars.items():
                if value:
                    formatted += [env_string, "=".join((name, str(value)))]
                else:
                    formatted += [env_string, name]
        return formatted

    def format_launch_args(self) -> t.List[str]:
        """Return a list of MPI-standard formatted run arguments

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
        """Set the launch arguments"""
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


@dispatch(to_launcher=ShellLauncher)
class MpiArgBuilder(_BaseMPIArgBuilder):
    def launcher_str(self) -> str:
        """Get the string representation of the launcher"""
        return LauncherType.Mpirun.value

    def finalize(
        self,
        exe: ExecutableLike,
        env: t.Mapping[str, str | None],
        job_execution_path: str,
    ) -> t.Tuple[t.Sequence[str], str]:
        return (
            "mpirun",
            *self.format_launch_args(),
            "--",
            *exe.as_program_arguments(),
        ), job_execution_path


@dispatch(to_launcher=ShellLauncher)
class MpiexecArgBuilder(_BaseMPIArgBuilder):
    def launcher_str(self) -> str:
        """Get the string representation of the launcher"""
        return LauncherType.Mpiexec.value

    def finalize(
        self,
        exe: ExecutableLike,
        env: t.Mapping[str, str | None],
        job_execution_path: str,
    ) -> t.Tuple[t.Sequence[str], str]:
        return (
            "mpiexec",
            *self.format_launch_args(),
            "--",
            *exe.as_program_arguments(),
        ), job_execution_path


@dispatch(to_launcher=ShellLauncher)
class OrteArgBuilder(_BaseMPIArgBuilder):
    def launcher_str(self) -> str:
        """Get the string representation of the launcher"""
        return LauncherType.Orterun.value

    def finalize(
        self,
        exe: ExecutableLike,
        env: t.Mapping[str, str | None],
        job_execution_path: str,
    ) -> t.Tuple[t.Sequence[str], str]:
        return (
            "orterun",
            *self.format_launch_args(),
            "--",
            *exe.as_program_arguments(),
        ), job_execution_path
