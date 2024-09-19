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

import os
import pathlib
import re
import subprocess
import typing as t

from smartsim._core.arguments.shell import ShellLaunchArguments
from smartsim._core.dispatch import EnvironMappingType, dispatch
from smartsim._core.shell.shell_launcher import ShellLauncher, ShellLauncherCommand
from smartsim.log import get_logger

from ...common import set_check_input
from ...launch_command import LauncherType

logger = get_logger(__name__)


def _as_srun_command(
    args: ShellLaunchArguments,
    exe: t.Sequence[str],
    path: pathlib.Path,
    env: EnvironMappingType,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
) -> ShellLauncherCommand:
    command_tuple = (
        "srun",
        *(args.format_launch_args() or ()),
        f"--output={stdout_path}",
        f"--error={stderr_path}",
        "--",
        *exe,
    )
    return ShellLauncherCommand(
        env, path, subprocess.DEVNULL, subprocess.DEVNULL, command_tuple
    )


@dispatch(with_format=_as_srun_command, to_launcher=ShellLauncher)
class SlurmLaunchArguments(ShellLaunchArguments):
    def launcher_str(self) -> str:
        """Get the string representation of the launcher

        :returns: The string representation of the launcher
        """
        return LauncherType.Slurm.value

    def _reserved_launch_args(self) -> set[str]:
        """Return reserved launch arguments.

        :returns: The set of reserved launcher arguments
        """
        return {"chdir", "D"}

    def set_nodes(self, nodes: int) -> None:
        """Set the number of nodes

        Effectively this is setting: ``srun --nodes <num_nodes>``

        :param nodes: nodes to launch on
        :return: launcher argument
        """
        self.set("nodes", str(nodes))

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        This sets ``--nodelist``

        :param host_list: hosts to launch on
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        elif not isinstance(host_list, list):
            raise TypeError("host_list argument must be a string or list of strings")
        elif not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self.set("nodelist", ",".join(host_list))

    def set_hostlist_from_file(self, file_path: str) -> None:
        """Use the contents of a file to set the node list

        This sets ``--nodefile``

        :param file_path: Path to the nodelist file
        """
        self.set("nodefile", file_path)

    def set_excluded_hosts(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify a list of hosts to exclude for launching this job

        :param host_list: hosts to exclude
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self.set("exclude", ",".join(host_list))

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of cpus to use per task

        This sets ``--cpus-per-task``

        :param num_cpus: number of cpus to use per task
        """
        self.set("cpus-per-task", str(cpus_per_task))

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        This sets ``--ntasks``

        :param tasks: number of tasks
        """
        self.set("ntasks", str(tasks))

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks for this job

        This sets ``--ntasks-per-node``

        :param tasks_per_node: number of tasks per node
        """
        self.set("ntasks-per-node", str(tasks_per_node))

    def set_cpu_bindings(self, bindings: t.Union[int, t.List[int]]) -> None:
        """Bind by setting CPU masks on tasks

        This sets ``--cpu-bind`` using the ``map_cpu:<list>`` option

        :param bindings: List specifing the cores to which MPI processes are bound
        """
        if isinstance(bindings, int):
            bindings = [bindings]
        self.set("cpu_bind", "map_cpu:" + ",".join(str(num) for num in bindings))

    def set_memory_per_node(self, memory_per_node: int) -> None:
        """Specify the real memory required per node

        This sets ``--mem`` in megabytes

        :param memory_per_node: Amount of memory per node in megabytes
        """
        self.set("mem", f"{memory_per_node}M")

    def set_executable_broadcast(self, dest_path: str) -> None:
        """Copy executable file to allocated compute nodes

        This sets ``--bcast``

        :param dest_path: Path to copy an executable file
        """
        self.set("bcast", dest_path)

    def set_node_feature(self, feature_list: t.Union[str, t.List[str]]) -> None:
        """Specify the node feature for this job

        This sets ``-C``

        :param feature_list: node feature to launch on
        :raises TypeError: if not str or list of str
        """
        if isinstance(feature_list, str):
            feature_list = [feature_list.strip()]
        elif not all(isinstance(feature, str) for feature in feature_list):
            raise TypeError("node_feature argument must be string or list of strings")
        self.set("C", ",".join(feature_list))

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        format = "HH:MM:SS"

        :param walltime: wall time
        """
        pattern = r"^\d{2}:\d{2}:\d{2}$"
        if walltime and re.match(pattern, walltime):
            self.set("time", str(walltime))
        else:
            raise ValueError("Invalid walltime format. Please use 'HH:MM:SS' format.")

    def set_het_group(self, het_group: t.Iterable[int]) -> None:
        """Set the heterogeneous group for this job

        this sets `--het-group`

        :param het_group: list of heterogeneous groups
        """
        het_size_env = os.getenv("SLURM_HET_SIZE")
        if het_size_env is None:
            msg = "Requested to set het group, but the allocation is not a het job"
            raise ValueError(msg)
        het_size = int(het_size_env)
        if any(group >= het_size for group in het_group):
            msg = (
                f"Het group {max(het_group)} requested, "
                f"but max het group in allocation is {het_size-1}"
            )
            raise ValueError(msg)
        self.set("het-group", ",".join(str(group) for group in het_group))

    def set_verbose_launch(self, verbose: bool) -> None:
        """Set the job to run in verbose mode

        This sets ``--verbose``

        :param verbose: Whether the job should be run verbosely
        """
        if verbose:
            self.set("verbose", None)
        else:
            self._launch_args.pop("verbose", None)

    def set_quiet_launch(self, quiet: bool) -> None:
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        """
        if quiet:
            self.set("quiet", None)
        else:
            self._launch_args.pop("quiet", None)

    def format_launch_args(self) -> t.List[str]:
        """Return a list of slurm formatted launch arguments

        :return: list of slurm arguments for these settings
        """
        formatted = []
        for key, value in self._launch_args.items():
            short_arg = bool(len(str(key)) == 1)
            prefix = "-" if short_arg else "--"
            if not value:
                formatted += [prefix + key]
            else:
                if short_arg:
                    formatted += [prefix + key, str(value)]
                else:
                    formatted += ["=".join((prefix + key, str(value)))]
        return formatted

    def format_env_vars(self, env_vars: t.Mapping[str, str | None]) -> list[str]:
        """Build bash compatible environment variable string for Slurm

        :returns: the formatted string of environment variables
        """
        self._check_env_vars(env_vars)
        return [f"{k}={v}" for k, v in env_vars.items() if "," not in str(v)]

    def format_comma_sep_env_vars(
        self, env_vars: t.Dict[str, t.Optional[str]]
    ) -> t.Union[t.Tuple[str, t.List[str]], None]:
        """Build environment variable string for Slurm

        Slurm takes exports in comma separated lists
        the list starts with all as to not disturb the rest of the environment
        for more information on this, see the slurm documentation for srun

        :param env_vars: An environment mapping
        :returns: the formatted string of environment variables
        """
        self._check_env_vars(env_vars)
        exportable_env, compound_env, key_only = [], [], []

        for k, v in env_vars.items():
            kvp = f"{k}={v}"

            if "," in str(v):
                key_only.append(k)
                compound_env.append(kvp)
            else:
                exportable_env.append(kvp)

        # Append keys to exportable KVPs, e.g. `--export x1=v1,KO1,KO2`
        fmt_exported_env = ",".join(v for v in exportable_env + key_only)

        return fmt_exported_env, compound_env

    def _check_env_vars(self, env_vars: t.Mapping[str, str | None]) -> None:
        """Warn a user trying to set a variable which is set in the environment

        Given Slurm's env var precedence, trying to export a variable which is already
        present in the environment will not work.
        """
        for k, v in env_vars.items():
            if "," not in str(v):
                # If a variable is defined, it will take precedence over --export
                # we warn the user
                preexisting_var = os.environ.get(k, None)
                if preexisting_var is not None and preexisting_var != v:
                    msg = (
                        f"Variable {k} is set to {preexisting_var} in current "
                        "environment. If the job is running in an interactive "
                        f"allocation, the value {v} will not be set. Please "
                        "consider removing the variable from the environment "
                        "and re-run the experiment."
                    )
                    logger.warning(msg)

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
