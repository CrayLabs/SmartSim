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

import datetime
import os
import typing as t

from ..error import SSUnsupportedError
from ..log import get_logger
from .base import BatchSettings, RunSettings

logger = get_logger(__name__)


class SrunSettings(RunSettings):
    def __init__(
        self,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.List[str]]] = None,
        run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        alloc: t.Optional[str] = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialize run parameters for a slurm job with ``srun``

        ``SrunSettings`` should only be used on Slurm based systems.

        If an allocation is specified, the instance receiving these run
        parameters will launch on that allocation.

        :param exe: executable to run
        :param exe_args: executable arguments
        :param run_args: srun arguments without dashes
        :param env_vars: environment variables for job
        :param alloc: allocation ID if running on existing alloc
        """
        super().__init__(
            exe,
            exe_args,
            run_command="srun",
            run_args=run_args,
            env_vars=env_vars,
            **kwargs,
        )
        self.alloc = alloc
        self.mpmd: t.List[RunSettings] = []

    reserved_run_args = {"chdir", "D"}

    def set_nodes(self, nodes: int) -> None:
        """Set the number of nodes

        Effectively this is setting: ``srun --nodes <num_nodes>``

        :param nodes: number of nodes to run with
        """
        self.run_args["nodes"] = int(nodes)

    def make_mpmd(self, settings: RunSettings) -> None:
        """Make a mpmd workload by combining two ``srun`` commands

        This connects the two settings to be executed with a single
        Model instance

        :param settings: SrunSettings instance
        """
        if self.colocated_db_settings:
            raise SSUnsupportedError(
                "Colocated models cannot be run as a mpmd workload"
            )
        if self.container:
            raise SSUnsupportedError(
                "Containerized MPMD workloads are not yet supported."
            )
        if os.getenv("SLURM_HET_SIZE") is not None:
            raise ValueError(
                "Slurm does not support MPMD workloads in heterogeneous jobs."
            )
        self.mpmd.append(settings)

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        This sets ``--nodelist``

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

        This sets ``--nodefile``

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

        This sets ``--cpus-per-task``

        :param num_cpus: number of cpus to use per task
        """
        self.run_args["cpus-per-task"] = int(cpus_per_task)

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        This sets ``--ntasks``

        :param tasks: number of tasks
        """
        self.run_args["ntasks"] = int(tasks)

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks for this job

        This sets ``--ntasks-per-node``

        :param tasks_per_node: number of tasks per node
        """
        self.run_args["ntasks-per-node"] = int(tasks_per_node)

    def set_cpu_bindings(self, bindings: t.Union[int, t.List[int]]) -> None:
        """Bind by setting CPU masks on tasks

        This sets ``--cpu-bind`` using the ``map_cpu:<list>`` option

        :param bindings: List specifing the cores to which MPI processes are bound
        """
        if isinstance(bindings, int):
            bindings = [bindings]
        self.run_args["cpu_bind"] = "map_cpu:" + ",".join(
            str(int(num)) for num in bindings
        )

    def set_memory_per_node(self, memory_per_node: int) -> None:
        """Specify the real memory required per node

        This sets ``--mem`` in megabytes

        :param memory_per_node: Amount of memory per node in megabytes
        """
        self.run_args["mem"] = f"{int(memory_per_node)}M"

    def set_verbose_launch(self, verbose: bool) -> None:
        """Set the job to run in verbose mode

        This sets ``--verbose``

        :param verbose: Whether the job should be run verbosely
        """
        if verbose:
            self.run_args["verbose"] = None
        else:
            self.run_args.pop("verbose", None)

    def set_quiet_launch(self, quiet: bool) -> None:
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        """
        if quiet:
            self.run_args["quiet"] = None
        else:
            self.run_args.pop("quiet", None)

    def set_broadcast(self, dest_path: t.Optional[str] = None) -> None:
        """Copy executable file to allocated compute nodes

        This sets ``--bcast``

        :param dest_path: Path to copy an executable file
        """
        self.run_args["bcast"] = dest_path

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
        self.run_args["C"] = ",".join(feature_list)

    @staticmethod
    def _fmt_walltime(hours: int, minutes: int, seconds: int) -> str:
        """Convert hours, minutes, and seconds into valid walltime format

        Converts time to format HH:MM:SS

        :param hours: number of hours to run job
        :param minutes: number of minutes to run job
        :param seconds: number of seconds to run job
        :returns: Formatted walltime
        """
        return fmt_walltime(hours, minutes, seconds)

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        format = "HH:MM:SS"

        :param walltime: wall time
        """
        self.run_args["time"] = str(walltime)

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
        if self.mpmd:
            msg = "Slurm does not support MPMD workloads in heterogeneous jobs\n"
            raise ValueError(msg)
        msg = (
            "Support for heterogeneous groups is an experimental feature, "
            "please report any unexpected behavior to SmartSim developers "
            "by opening an issue on https://github.com/CrayLabs/SmartSim/issues"
        )
        if any(group >= het_size for group in het_group):
            msg = (
                f"Het group {max(het_group)} requested, "
                f"but max het group in allocation is {het_size-1}"
            )
            raise ValueError(msg)
        logger.warning(msg)
        self.run_args["het-group"] = ",".join(str(group) for group in het_group)

    def format_run_args(self) -> t.List[str]:
        """Return a list of slurm formatted run arguments

        :return: list of slurm arguments for these settings
        """
        # add additional slurm arguments based on key length
        opts = []
        for opt, value in self.run_args.items():
            short_arg = bool(len(str(opt)) == 1)
            prefix = "-" if short_arg else "--"
            if not value:
                opts += [prefix + opt]
            else:
                if short_arg:
                    opts += [prefix + opt, str(value)]
                else:
                    opts += ["=".join((prefix + opt, str(value)))]
        return opts

    def check_env_vars(self) -> None:
        """Warn a user trying to set a variable which is set in the environment

        Given Slurm's env var precedence, trying to export a variable which is already
        present in the environment will not work.
        """
        for k, v in self.env_vars.items():
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

    def format_env_vars(self) -> t.List[str]:
        """Build bash compatible environment variable string for Slurm

        :returns: the formatted string of environment variables
        """
        self.check_env_vars()
        return [f"{k}={v}" for k, v in self.env_vars.items() if "," not in str(v)]

    def format_comma_sep_env_vars(self) -> t.Tuple[str, t.List[str]]:
        """Build environment variable string for Slurm

        Slurm takes exports in comma separated lists
        the list starts with all as to not disturb the rest of the environment
        for more information on this, see the slurm documentation for srun

        :returns: the formatted string of environment variables
        """
        self.check_env_vars()
        exportable_env, compound_env, key_only = [], [], []

        for k, v in self.env_vars.items():
            kvp = f"{k}={v}"

            if "," in str(v):
                key_only.append(k)
                compound_env.append(kvp)
            else:
                exportable_env.append(kvp)

        # Append keys to exportable KVPs, e.g. `--export x1=v1,KO1,KO2`
        fmt_exported_env = ",".join(v for v in exportable_env + key_only)

        for mpmd in self.mpmd:
            compound_mpmd_env = {
                k: v for k, v in mpmd.env_vars.items() if "," in str(v)
            }
            compound_mpmd_fmt = {f"{k}={v}" for k, v in compound_mpmd_env.items()}
            compound_env.extend(compound_mpmd_fmt)

        return fmt_exported_env, compound_env


def fmt_walltime(hours: int, minutes: int, seconds: int) -> str:
    """Helper function walltime format conversion

    Converts time to format HH:MM:SS

    :param hours: number of hours to run job
    :param minutes: number of minutes to run job
    :param seconds: number of seconds to run job
    :returns: Formatted walltime
    """
    delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    fmt_str = str(delta)
    if delta.seconds // 3600 < 10:
        fmt_str = "0" + fmt_str
    return fmt_str


class SbatchSettings(BatchSettings):
    def __init__(
        self,
        nodes: t.Optional[int] = None,
        time: str = "",
        account: t.Optional[str] = None,
        batch_args: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ) -> None:
        """Specify run parameters for a Slurm batch job

        Slurm `sbatch` arguments can be written into ``batch_args``
        as a dictionary. e.g. {'ntasks': 1}

        If the argument doesn't have a parameter, put `None`
        as the value. e.g. {'exclusive': None}

        Initialization values provided (nodes, time, account)
        will overwrite the same arguments in ``batch_args`` if present

        :param nodes: number of nodes
        :param time: walltime for job, e.g. "10:00:00" for 10 hours
        :param account: account for job
        :param batch_args: extra batch arguments
        """
        super().__init__(
            "sbatch",
            batch_args=batch_args,
            nodes=nodes,
            account=account,
            time=time,
            **kwargs,
        )

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        format = "HH:MM:SS"

        :param walltime: wall time
        """
        # TODO check for formatting here
        if walltime:
            self.batch_args["time"] = walltime

    def set_nodes(self, num_nodes: int) -> None:
        """Set the number of nodes for this batch job

        :param num_nodes: number of nodes
        """
        if num_nodes:
            self.batch_args["nodes"] = str(int(num_nodes))

    def set_account(self, account: str) -> None:
        """Set the account for this batch job

        :param account: account id
        """
        if account:
            self.batch_args["account"] = account

    def set_partition(self, partition: str) -> None:
        """Set the partition for the batch job

        :param partition: partition name
        """
        self.batch_args["partition"] = str(partition)

    def set_queue(self, queue: str) -> None:
        """alias for set_partition

        Sets the partition for the slurm batch job

        :param queue: the partition to run the batch job on
        """
        if queue:
            self.set_partition(queue)

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of cpus to use per task

        This sets ``--cpus-per-task``

        :param num_cpus: number of cpus to use per task
        """
        self.batch_args["cpus-per-task"] = str(int(cpus_per_task))

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
        self.batch_args["nodelist"] = ",".join(host_list)

    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Sbatch
        """
        opts = []
        # TODO add restricted here
        for opt, value in self.batch_args.items():
            # attach "-" prefix if argument is 1 character otherwise "--"
            short_arg = bool(len(str(opt)) == 1)
            prefix = "-" if short_arg else "--"

            if not value:
                opts += [prefix + opt]
            else:
                if short_arg:
                    opts += [prefix + opt, str(value)]
                else:
                    opts += ["=".join((prefix + opt, str(value)))]
        return opts
