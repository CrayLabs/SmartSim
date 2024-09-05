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

# pylint: disable=too-many-lines

import itertools
import os.path as osp
import shutil
import sys
import typing as t
from os import environ, getcwd, getenv
from shlex import split as sh_split

import psutil

from smartsim.entity._mock import Mock

from .._core.utils.helpers import is_valid_cmd, unpack_fs_identifier
from .._core.utils.network import get_ip_from_host
from .._core.utils.shell import execute_cmd
from ..entity import FSNode, TelemetryConfiguration
from ..error import SmartSimError, SSDBFilesNotParseable, SSUnsupportedError
from ..log import get_logger
from ..servertype import CLUSTERED, STANDALONE
from ..settings import (
    AprunSettings,
    BatchSettings,
    BsubBatchSettings,
    JsrunSettings,
    MpiexecSettings,
    MpirunSettings,
    OrterunSettings,
    PalsMpiexecSettings,
    QsubBatchSettings,
    RunSettings,
    SbatchSettings,
    SrunSettings,
    create_batch_settings,
    create_run_settings,
)
from ..wlm import detect_launcher

logger = get_logger(__name__)


class Client(Mock):
    """Mock Client"""


class ConfigOptions(Mock):
    """Mock ConfigOptions"""


def fs_is_active():
    return False


by_launcher: t.Dict[str, t.List[str]] = {
    "dragon": [""],
    "slurm": ["srun", "mpirun", "mpiexec"],
    "pbs": ["aprun", "mpirun", "mpiexec"],
    "pals": ["mpiexec"],
    "lsf": ["jsrun"],
    "local": [""],
    "sge": ["mpirun", "mpiexec", "orterun"],
}


def _detect_command(launcher: str) -> str:
    if launcher in by_launcher:
        for cmd in by_launcher[launcher]:
            if launcher in ["local", "dragon"]:
                return cmd
            if is_valid_cmd(cmd):
                return cmd
    msg = (
        "Could not automatically detect a run command to use for launcher "
        f"{launcher}\nSearched for and could not find the following "
        f"commands: {by_launcher[launcher]}"
    )
    raise SmartSimError(msg)


def _autodetect(launcher: str, run_command: str) -> t.Tuple[str, str]:
    """Automatically detect the launcher and run command to use"""
    if launcher == "auto":
        launcher = detect_launcher()

    if run_command == "auto":
        run_command = _detect_command(launcher)

    return launcher, run_command


def _check_run_command(launcher: str, run_command: str) -> None:
    """Check that the run command is supported by the launcher"""
    if run_command not in by_launcher[launcher]:
        msg = (
            f"Run command {run_command} is not supported on launcher {launcher}\n"
            + "Supported run commands for the given launcher are: "
            + f"{by_launcher[launcher]}"
        )
        raise SmartSimError(msg)


def _get_single_command(
    run_command: str, launcher: str, batch: bool, single_cmd: bool
) -> bool:
    if not single_cmd:
        return single_cmd

    if launcher == "dragon":
        return False

    if run_command == "srun" and getenv("SLURM_HET_SIZE") is not None:
        msg = (
            "srun can not launch an FeatureStore with single_cmd=True in "
            + "a hetereogeneous job. Automatically switching to single_cmd=False."
        )
        logger.info(msg)
        return False

    if not batch:
        return single_cmd

    if run_command == "aprun":
        msg = (
            "aprun can not launch an FeatureStore with batch=True and "
            + "single_cmd=True. Automatically switching to single_cmd=False."
        )
        logger.info(msg)
        return False

    return single_cmd


def _check_local_constraints(launcher: str, batch: bool) -> None:
    """Check that the local launcher is not launched with invalid batch config"""
    if launcher == "local" and batch:
        msg = "Local FeatureStore can not be launched with batch=True"
        raise SmartSimError(msg)


# pylint: disable-next=too-many-public-methods
class FeatureStore:
    """The FeatureStore is an in-memory database that can be launched
    alongside entities in SmartSim. Data can be transferred between
    entities by using one of the Python, C, C++ or Fortran clients
    within an entity.
    """

    def __init__(
        self,
        path: t.Optional[str] = getcwd(),
        port: int = 6379,
        interface: t.Union[str, t.List[str]] = "lo",
        launcher: str = "local",
        run_command: str = "auto",
        fs_nodes: int = 1,
        batch: bool = False,
        hosts: t.Optional[t.Union[t.List[str], str]] = None,
        account: t.Optional[str] = None,
        time: t.Optional[str] = None,
        alloc: t.Optional[str] = None,
        single_cmd: bool = False,
        *,
        threads_per_queue: t.Optional[int] = None,
        inter_op_threads: t.Optional[int] = None,
        intra_op_threads: t.Optional[int] = None,
        fs_identifier: str = "featurestore",
        **kwargs: t.Any,
    ) -> None:
        """Initialize an ``FeatureStore`` reference for local launch

        Extra configurations

        :param path: path to location of ``FeatureStore`` directory
        :param port: TCP/IP port
        :param interface: network interface(s)
        :param launcher: type of launcher being used, options are "slurm", "pbs",
                         "lsf", or "local". If set to "auto",
                         an attempt will be made to find an available launcher
                         on the system.
        :param run_command: specify launch binary or detect automatically
        :param fs_nodes: number of feature store shards
        :param batch: run as a batch workload
        :param hosts: specify hosts to launch on
        :param account: account to run batch on
        :param time: walltime for batch 'HH:MM:SS' format
        :param alloc: allocation to launch feature store on
        :param single_cmd: run all shards with one (MPMD) command
        :param threads_per_queue: threads per GPU device
        :param inter_op_threads: threads across CPU operations
        :param intra_op_threads: threads per CPU operation
        :param fs_identifier: an identifier to distinguish this FeatureStore in
            multiple-feature store experiments
        """
        self.launcher, self.run_command = _autodetect(launcher, run_command)
        _check_run_command(self.launcher, self.run_command)
        _check_local_constraints(self.launcher, batch)
        single_cmd = _get_single_command(
            self.run_command, self.launcher, batch, single_cmd
        )
        self.ports: t.List[int] = []
        self._hosts: t.List[str] = []
        self._user_hostlist: t.List[str] = []
        if isinstance(interface, str):
            interface = [interface]
        self._interfaces = interface
        self._check_network_interface()
        self.queue_threads = threads_per_queue
        self.inter_threads = inter_op_threads
        self.intra_threads = intra_op_threads
        self._telemetry_cfg = TelemetryConfiguration()

        gpus_per_shard: t.Optional[int] = None
        cpus_per_shard: t.Optional[int] = None
        if self.launcher == "lsf":
            gpus_per_shard = int(kwargs.pop("gpus_per_shard", 0))
            cpus_per_shard = int(kwargs.pop("cpus_per_shard", 4))
        super().__init__(
            name=fs_identifier,
            path=str(path),
            port=port,
            interface=interface,
            fs_nodes=fs_nodes,
            batch=batch,
            launcher=self.launcher,
            run_command=self.run_command,
            alloc=alloc,
            single_cmd=single_cmd,
            gpus_per_shard=gpus_per_shard,
            cpus_per_shard=cpus_per_shard,
            threads_per_queue=threads_per_queue,
            inter_op_threads=inter_op_threads,
            intra_op_threads=intra_op_threads,
            **kwargs,
        )

        if self.launcher != "local":
            self.batch_settings = self._build_batch_settings(
                fs_nodes,
                alloc or "",
                batch,
                account or "",
                time or "",
                launcher=self.launcher,
                **kwargs,
            )
            if hosts:
                self.set_hosts(hosts)
            elif not hosts:
                mpilike = run_command in ["mpirun", "mpiexec", "orterun"]
                if mpilike and not self._mpi_has_sge_support():
                    raise SmartSimError(
                        "hosts argument required when launching "
                        f"{type(self).__name__} with mpirun"
                    )
            self._reserved_run_args: t.Dict[t.Type[RunSettings], t.List[str]] = {}
            self._reserved_batch_args: t.Dict[t.Type[BatchSettings], t.List[str]] = {}
            self._fill_reserved()

    def _mpi_has_sge_support(self) -> bool:
        """Check if MPI command supports SGE

        If the run command is mpirun, mpiexec, or orterun, there is a possibility
        that the user is using OpenMPI with SGE grid support. In this case, hosts
        do not need to be set.

        :returns: bool
        """

        if self.run_command in ["mpirun", "orterun", "mpiexec"]:
            if shutil.which("ompi_info"):
                _, output, _ = execute_cmd(["ompi_info"])
                return "gridengine" in output
        return False

    @property
    def fs_identifier(self) -> str:
        """Return the FS identifier, which is common to a FS and all of its nodes

        :return: FS identifier
        """
        return self.name

    @property
    def num_shards(self) -> int:
        """Return the number of FS shards contained in the FeatureStore.
        This might differ from the number of ``FSNode`` objects, as each
        ``FSNode`` may start more than one shard (e.g. with MPMD).

        :returns: the number of FS shards contained in the FeatureStore
        """
        return sum(node.num_shards for node in self.entities)

    @property
    def fs_nodes(self) -> int:
        """Read only property for the number of nodes an ``FeatureStore`` is
        launched across. Notice that SmartSim currently assumes that each shard
        will be launched on its own node. Therefore this property is currently
        an alias to the ``num_shards`` attribute.

        :returns: Number of feature store nodes
        """
        return self.num_shards

    @property
    def hosts(self) -> t.List[str]:
        """Return the hostnames of FeatureStore instance hosts

        Note that this will only be populated after the FeatureStore
        has been launched by SmartSim.

        :return: the hostnames of FeatureStore instance hosts
        """
        if not self._hosts:
            self._hosts = self._get_fs_hosts()
        return self._hosts

    @property
    def telemetry(self) -> TelemetryConfiguration:
        """Return the telemetry configuration for this entity.

        :returns: configuration of telemetry for this entity
        """
        return self._telemetry_cfg

    def reset_hosts(self) -> None:
        """Clear hosts or reset them to last user choice"""
        for node in self.entities:
            node.clear_hosts()
        self._hosts = []
        # This is only needed on LSF
        if self._user_hostlist:
            self.set_hosts(self._user_hostlist)

    def remove_stale_files(self) -> None:
        """Can be used to remove feature store files of a previous launch"""

        for fs in self.entities:
            fs.remove_stale_fsnode_files()

    def get_address(self) -> t.List[str]:
        """Return feature store addresses

        :return: addresses

        :raises SmartSimError: If feature store address cannot be found or is not active
        """
        if not self._hosts:
            raise SmartSimError("Could not find feature store address")
        if not self.is_active():
            raise SmartSimError("Feature store is not active")
        return self._get_address()

    def _get_address(self) -> t.List[str]:
        return [
            f"{host}:{port}"
            for host, port in itertools.product(self._hosts, self.ports)
        ]

    def is_active(self) -> bool:
        """Check if the feature store is active

        :return: True if feature store is active, False otherwise
        """
        try:
            hosts = self.hosts
        except SSDBFilesNotParseable:
            return False
        return fs_is_active(hosts, self.ports, self.num_shards)

    @property
    def checkpoint_file(self) -> str:
        """Get the path to the checkpoint file for this Feature Store

        :return: Path to the checkpoint file if it exists, otherwise a None
        """
        return osp.join(self.path, "smartsim_db.dat")

    def set_cpus(self, num_cpus: int) -> None:
        """Set the number of CPUs available to each feature store shard

        This effectively will determine how many cpus can be used for
        compute threads, background threads, and network I/O.

        :param num_cpus: number of cpus to set
        """
        if self.batch:
            if self.launcher == "pbs":
                if hasattr(self, "batch_settings") and self.batch_settings:
                    if hasattr(self.batch_settings, "set_ncpus"):
                        self.batch_settings.set_ncpus(num_cpus)
            if self.launcher == "slurm":
                if hasattr(self, "batch_settings") and self.batch_settings:
                    if hasattr(self.batch_settings, "set_cpus_per_task"):
                        self.batch_settings.set_cpus_per_task(num_cpus)

        for fs in self.entities:
            fs.run_settings.set_cpus_per_task(num_cpus)
            if fs.is_mpmd and hasattr(fs.run_settings, "mpmd"):
                for mpmd in fs.run_settings.mpmd:
                    mpmd.set_cpus_per_task(num_cpus)

    def set_walltime(self, walltime: str) -> None:
        """Set the batch walltime of the FeatureStore

        Note: This will only effect FeatureStores launched as a batch

        :param walltime: amount of time e.g. 10 hours is 10:00:00
        :raises SmartSimError: if FeatureStore isn't launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set walltime")

        if hasattr(self, "batch_settings") and self.batch_settings:
            self.batch_settings.set_walltime(walltime)

    def set_hosts(self, host_list: t.Union[t.List[str], str]) -> None:
        """Specify the hosts for the ``FeatureStore`` to launch on

        :param host_list: list of host (compute node names)
        :raises TypeError: if wrong type
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        self._user_hostlist = host_list.copy()
        # TODO check length
        if self.batch and hasattr(self, "batch_settings") and self.batch_settings:
            self.batch_settings.set_hostlist(host_list)

        if self.launcher == "lsf":
            for fs in self.entities:
                fs.set_hosts(host_list)
        elif (
            self.launcher == "pals"
            and isinstance(self.entities[0].run_settings, PalsMpiexecSettings)
            and self.entities[0].is_mpmd
        ):
            # In this case, --hosts is a global option, set it to first run command
            self.entities[0].run_settings.set_hostlist(host_list)
        else:
            for host, fs in zip(host_list, self.entities):
                if isinstance(fs.run_settings, AprunSettings):
                    if not self.batch:
                        fs.run_settings.set_hostlist([host])
                else:
                    fs.run_settings.set_hostlist([host])

                if fs.is_mpmd and hasattr(fs.run_settings, "mpmd"):
                    for i, mpmd_runsettings in enumerate(fs.run_settings.mpmd, 1):
                        mpmd_runsettings.set_hostlist(host_list[i])

    def set_batch_arg(self, arg: str, value: t.Optional[str] = None) -> None:
        """Set a batch argument the FeatureStore should launch with

        Some commonly used arguments such as --job-name are used
        by SmartSim and will not be allowed to be set.

        :param arg: batch argument to set e.g. "exclusive"
        :param value: batch param - set to None if no param value
        :raises SmartSimError: if FeatureStore not launching as batch
        """
        if not hasattr(self, "batch_settings") or not self.batch_settings:
            raise SmartSimError("Not running as batch, cannot set batch_arg")

        if arg in self._reserved_batch_args[type(self.batch_settings)]:
            logger.warning(
                f"Can not set batch argument {arg}: "
                "it is a reserved keyword in FeatureStore"
            )
        else:
            self.batch_settings.batch_args[arg] = value

    def set_run_arg(self, arg: str, value: t.Optional[str] = None) -> None:
        """Set a run argument the FeatureStore should launch
        each node with (it will be passed to `jrun`)

        Some commonly used arguments are used
        by SmartSim and will not be allowed to be set.
        For example, "n", "N", etc.

        :param arg: run argument to set
        :param value: run parameter - set to None if no parameter value
        """
        if arg in self._reserved_run_args[type(self.entities[0].run_settings)]:
            logger.warning(
                f"Can not set batch argument {arg}: "
                "it is a reserved keyword in FeatureStore"
            )
        else:
            for fs in self.entities:
                fs.run_settings.run_args[arg] = value
                if fs.is_mpmd and hasattr(fs.run_settings, "mpmd"):
                    for mpmd in fs.run_settings.mpmd:
                        mpmd.run_args[arg] = value

    def enable_checkpoints(self, frequency: int) -> None:
        """Sets the feature store's save configuration to save the fs every 'frequency'
        seconds given that at least one write operation against the fs occurred in
        that time. E.g., if `frequency` is 900, then the feature store will save to disk
        after 900 seconds if there is at least 1 change to the dataset.

        :param frequency: the given number of seconds before the FS saves
        """
        self.set_fs_conf("save", f"{frequency} 1")

    def set_max_memory(self, mem: str) -> None:
        """Sets the max memory configuration. By default there is no memory limit.
        Setting max memory to zero also results in no memory limit. Once a limit is
        surpassed, keys will be removed according to the eviction strategy. The
        specified memory size is case insensitive and supports the typical forms of:

        1k => 1000 bytes \n
        1kb => 1024 bytes \n
        1m => 1000000 bytes \n
        1mb => 1024*1024 bytes \n
        1g => 1000000000 bytes \n
        1gb => 1024*1024*1024 bytes

        :param mem: the desired max memory size e.g. 3gb
        :raises SmartSimError: If 'mem' is an invalid memory value
        :raises SmartSimError: If feature store is not active
        """
        self.set_fs_conf("maxmemory", mem)

    def set_eviction_strategy(self, strategy: str) -> None:
        """Sets how the feature store will select what to remove when
        'maxmemory' is reached. The default is noeviction.

        :param strategy: The max memory policy to use
            e.g. "volatile-lru", "allkeys-lru", etc.
        :raises SmartSimError: If 'strategy' is an invalid maxmemory policy
        :raises SmartSimError: If feature store is not active
        """
        self.set_fs_conf("maxmemory-policy", strategy)

    def set_max_clients(self, clients: int = 50_000) -> None:
        """Sets the max number of connected clients at the same time.
        When the number of FS shards contained in the feature store is
        more than two, then every node will use two connections, one
        incoming and another outgoing.

        :param clients: the maximum number of connected clients
        """
        self.set_fs_conf("maxclients", str(clients))

    def set_max_message_size(self, size: int = 1_073_741_824) -> None:
        """Sets the feature store's memory size limit for bulk requests,
        which are elements representing single strings. The default
        is 1 gigabyte. Message size must be greater than or equal to 1mb.
        The specified memory size should be an integer that represents
        the number of bytes. For example, to set the max message size
        to 1gb, use 1024*1024*1024.

        :param size: maximum message size in bytes
        """
        self.set_fs_conf("proto-max-bulk-len", str(size))

    def set_fs_conf(self, key: str, value: str) -> None:
        """Set any valid configuration at runtime without the need
        to restart the feature store. All configuration parameters
        that are set are immediately loaded by the feature store and
        will take effect starting with the next command executed.

        :param key: the configuration parameter
        :param value: the feature store configuration parameter's new value
        """
        if self.is_active():
            addresses = []
            for host in self.hosts:
                for port in self.ports:
                    addresses.append(":".join([get_ip_from_host(host), str(port)]))

            fs_name, name = unpack_fs_identifier(self.fs_identifier, "_")

            environ[f"SSDB{fs_name}"] = addresses[0]

            fs_type = CLUSTERED if self.num_shards > 2 else STANDALONE
            environ[f"SR_DB_TYPE{fs_name}"] = fs_type

            options = ConfigOptions.create_from_environment(name)
            client = Client(options)

            try:
                for address in addresses:
                    client.config_set(key, value, address)

            except TypeError:
                raise TypeError(
                    "Incompatible function arguments. The key and value used for "
                    "setting the feature store configurations must be strings."
                ) from None
        else:
            raise SmartSimError(
                "The SmartSim FeatureStore must be active in order to set the "
                "feature store's configurations."
            )

    @staticmethod
    def _build_batch_settings(
        fs_nodes: int,
        alloc: str,
        batch: bool,
        account: str,
        time: str,
        *,
        launcher: t.Optional[str] = None,
        **kwargs: t.Any,
    ) -> t.Optional[BatchSettings]:
        batch_settings = None

        if launcher is None:
            raise ValueError("Expected param `launcher` of type `str`")

        # enter this conditional if user has not specified an allocation to run
        # on or if user specified batch=False (alloc will be found through env)
        if not alloc and batch:
            batch_settings = create_batch_settings(
                launcher, nodes=fs_nodes, time=time, account=account, **kwargs
            )

        return batch_settings

    def _build_run_settings(
        self,
        exe: str,
        exe_args: t.List[t.List[str]],
        *,
        run_args: t.Optional[t.Dict[str, t.Any]] = None,
        fs_nodes: int = 1,
        single_cmd: bool = True,
        **kwargs: t.Any,
    ) -> RunSettings:
        run_args = {} if run_args is None else run_args
        mpmd_nodes = single_cmd and fs_nodes > 1

        if mpmd_nodes:
            run_settings = create_run_settings(
                exe=exe, exe_args=exe_args[0], run_args=run_args.copy(), **kwargs
            )

            if self.launcher != "local":
                run_settings.set_tasks(1)

            for exe_arg in exe_args[1:]:
                mpmd_run_settings = create_run_settings(
                    exe=exe, exe_args=exe_arg, run_args=run_args.copy(), **kwargs
                )
                mpmd_run_settings.set_tasks(1)
                mpmd_run_settings.set_tasks_per_node(1)
                run_settings.make_mpmd(mpmd_run_settings)
        else:
            run_settings = create_run_settings(
                exe=exe, exe_args=exe_args[0], run_args=run_args.copy(), **kwargs
            )

            if self.launcher != "local":
                run_settings.set_tasks(1)

        if self.launcher != "local":
            run_settings.set_tasks_per_node(1)

        return run_settings

    @staticmethod
    def _build_run_settings_lsf(
        exe: str,
        exe_args: t.List[t.List[str]],
        *,
        run_args: t.Optional[t.Dict[str, t.Any]] = None,
        cpus_per_shard: t.Optional[int] = None,
        gpus_per_shard: t.Optional[int] = None,
        **_kwargs: t.Any,  # Needed to ensure no API break and do not want to
        # introduce that possibility, even if this method is
        # protected, without running the test suite.
    ) -> t.Optional[JsrunSettings]:
        run_args = {} if run_args is None else run_args
        erf_rs: t.Optional[JsrunSettings] = None

        if cpus_per_shard is None:
            raise ValueError("Expected an integer number of cpus per shard")
        if gpus_per_shard is None:
            raise ValueError("Expected an integer number of gpus per shard")

        # We always run the fs on cpus 0:cpus_per_shard-1
        # and gpus 0:gpus_per_shard-1
        for shard_id, args in enumerate(exe_args):
            host = shard_id
            run_args["launch_distribution"] = "packed"

            run_settings = JsrunSettings(exe, args, run_args=run_args.copy())
            run_settings.set_binding("none")

            # This makes sure output is written to featurestore_0.out,
            # featurestore_1.out, and so on
            run_settings.set_individual_output("_%t")

            erf_sets = {
                "rank": str(shard_id),
                "host": str(1 + host),
                "cpu": "{" + f"0:{cpus_per_shard}" + "}",
            }

            if gpus_per_shard > 1:  # pragma: no-cover
                erf_sets["gpu"] = f"{{0-{gpus_per_shard-1}}}"
            elif gpus_per_shard > 0:
                erf_sets["gpu"] = "{0}"

            run_settings.set_erf_sets(erf_sets)

            if not erf_rs:
                erf_rs = run_settings
                continue

            erf_rs.make_mpmd(run_settings)

        return erf_rs

    def _initialize_entities(
        self,
        *,
        fs_nodes: int = 1,
        single_cmd: bool = True,
        port: int = 6379,
        **kwargs: t.Any,
    ) -> None:
        fs_nodes = int(fs_nodes)
        if fs_nodes == 2:
            raise SSUnsupportedError("FeatureStore does not support clusters of size 2")

        if self.launcher == "local" and fs_nodes > 1:
            raise ValueError(
                "Local FeatureStore does not support multiple feature store shards"
            )

        mpmd_nodes = (single_cmd and fs_nodes > 1) or self.launcher == "lsf"

        if mpmd_nodes:
            self._initialize_entities_mpmd(
                fs_nodes=fs_nodes, single_cmd=single_cmd, port=port, **kwargs
            )
        else:
            cluster = fs_nodes >= 3

            for fs_id in range(fs_nodes):
                fs_node_name = "_".join((self.name, str(fs_id)))

                # create the exe_args list for launching multiple feature stores
                # per node. also collect port range for fsnode
                start_script_args = self._get_start_script_args(
                    fs_node_name, port, cluster
                )
                # if only launching 1 fs per command, we don't need a
                # list of exe args lists
                run_settings = self._build_run_settings(
                    sys.executable, [start_script_args], port=port, **kwargs
                )

                node = FSNode(
                    fs_node_name,
                    self.path,
                    exe=sys.executable,
                    exe_args=[start_script_args],
                    run_settings=run_settings,
                    ports=[port],
                    output_files=[fs_node_name + ".out"],
                    fs_identifier=self.fs_identifier,
                )
                self.entities.append(node)

            self.ports = [port]

    def _initialize_entities_mpmd(
        self, *, fs_nodes: int = 1, port: int = 6379, **kwargs: t.Any
    ) -> None:
        cluster = fs_nodes >= 3
        mpmd_node_name = self.name + "_0"
        exe_args_mpmd: t.List[t.List[str]] = []

        for fs_id in range(fs_nodes):
            fs_shard_name = "_".join((self.name, str(fs_id)))
            # create the exe_args list for launching multiple feature stores
            # per node. also collect port range for fsnode
            start_script_args = self._get_start_script_args(
                fs_shard_name, port, cluster
            )
            exe_args = " ".join(start_script_args)
            exe_args_mpmd.append(sh_split(exe_args))
        run_settings: t.Optional[RunSettings] = None
        if self.launcher == "lsf":
            run_settings = self._build_run_settings_lsf(
                sys.executable, exe_args_mpmd, fs_nodes=fs_nodes, port=port, **kwargs
            )
            output_files = [f"{self.name}_{fs_id}.out" for fs_id in range(fs_nodes)]
        else:
            run_settings = self._build_run_settings(
                sys.executable, exe_args_mpmd, fs_nodes=fs_nodes, port=port, **kwargs
            )
            output_files = [mpmd_node_name + ".out"]
        if not run_settings:
            raise ValueError(f"Could not build run settings for {self.launcher}")
        node = FSNode(
            mpmd_node_name,
            self.path,
            run_settings,
            [port],
            output_files,
            fs_identifier=self.fs_identifier,
        )
        self.entities.append(node)
        self.ports = [port]

    def _get_start_script_args(
        self, name: str, port: int, cluster: bool
    ) -> t.List[str]:
        cmd = [
            "-m",
            f"+name={name}",  # name of node
            f"+ifname={','.join(self._interfaces)}",  # pass interface to start script
        ]
        if cluster:
            cmd.append("+cluster")  # is the shard part of a cluster

        return cmd

    def _get_fs_hosts(self) -> t.List[str]:
        hosts = []
        for fs in self.entities:
            if not fs.is_mpmd:
                hosts.append(fs.host)
            else:
                hosts.extend(fs.hosts)
        return hosts

    def _check_network_interface(self) -> None:
        net_if_addrs = psutil.net_if_addrs()
        for interface in self._interfaces:
            if interface not in net_if_addrs and interface != "lo":
                available = list(net_if_addrs.keys())
                logger.warning(
                    f"{interface} is not a valid network interface on this node. \n"
                    "This could be because the head node doesn't have the same "
                    "networks, if so, ignore this."
                )
                logger.warning(f"Found network interfaces are: {available}")

    def _fill_reserved(self) -> None:
        """Fill the reserved batch and run arguments dictionaries"""

        mpi_like_settings = [
            MpirunSettings,
            MpiexecSettings,
            OrterunSettings,
            PalsMpiexecSettings,
        ]
        for settings in mpi_like_settings:
            self._reserved_run_args[settings] = [
                "np",
                "N",
                "c",
                "output-filename",
                "n",
                "wdir",
                "wd",
                "host",
            ]
        self._reserved_run_args[SrunSettings] = [
            "nodes",
            "N",
            "ntasks",
            "n",
            "ntasks-per-node",
            "output",
            "o",
            "error",
            "e",
            "job-name",
            "J",
            "jobid",
            "multi-prog",
            "w",
            "chdir",
            "D",
        ]
        self._reserved_run_args[AprunSettings] = [
            "pes",
            "n",
            "pes-per-node",
            "N",
            "l",
            "pes-per-numa-node",
            "S",
            "wdir",
        ]
        self._reserved_batch_args[SbatchSettings] = [
            "nodes",
            "N",
            "ntasks",
            "n",
            "ntasks-per-node",
            "output",
            "o",
            "error",
            "e",
            "job-name",
            "J",
            "jobid",
            "multi-prog",
            "w",
            "chdir",
            "D",
        ]
        self._reserved_batch_args[QsubBatchSettings] = ["e", "o", "N", "l"]
        self._reserved_run_args[JsrunSettings] = [
            "chdir",
            "h",
            "stdio_stdout",
            "o",
            "stdio_stderr",
            "k",
            "tasks_per_rs",
            "a",
            "np",
            "p",
            "cpu_per_rs",
            "c",
            "gpu_per_rs",
            "g",
            "latency_priority",
            "l",
            "memory_per_rs",
            "m",
            "nrs",
            "n",
            "rs_per_host",
            "r",
            "rs_per_socket",
            "K",
            "appfile",
            "f",
            "allocate_only",
            "A",
            "launch_node_task",
            "H",
            "use_reservation",
            "J",
            "use_resources",
            "bind",
            "b",
            "launch_distribution",
            "d",
        ]

        self._reserved_batch_args[BsubBatchSettings] = [
            "J",
            "o",
            "e",
            "m",
            "n",
            "nnodes",
        ]
