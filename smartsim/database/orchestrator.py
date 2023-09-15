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
import itertools
import psutil
import sys
import typing as t

from os import getcwd, getenv
from shlex import split as sh_split

from smartredis import Client
from smartredis.error import RedisReplyError

from .._core.config import CONFIG
from .._core.utils import db_is_active
from .._core.utils.helpers import is_valid_cmd
from .._core.utils.network import get_ip_from_host
from ..entity import DBNode, EntityList
from ..error import SmartSimError, SSConfigError, SSUnsupportedError
from ..log import get_logger
from ..settings.base import BatchSettings, RunSettings
from ..settings import (
    AprunSettings,
    BsubBatchSettings,
    CobaltBatchSettings,
    JsrunSettings,
    MpiexecSettings,
    MpirunSettings,
    OrterunSettings,
    PalsMpiexecSettings,
    QsubBatchSettings,
    SbatchSettings,
    SrunSettings,
)
from ..settings.settings import create_batch_settings, create_run_settings
from ..wlm import detect_launcher

logger = get_logger(__name__)


by_launcher: t.Dict[str, t.List[str]] = {
    "slurm": ["srun", "mpirun", "mpiexec"],
    "pbs": ["aprun", "mpirun", "mpiexec"],
    "pals": ["mpiexec"],
    "cobalt": ["aprun", "mpirun", "mpiexec"],
    "lsf": ["jsrun"],
    "local": [""],
}


def _detect_command(launcher: str) -> str:
    if launcher in by_launcher:
        for cmd in by_launcher[launcher]:
            if launcher == "local":
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


def _get_single_command(run_command: str, batch: bool, single_cmd: bool) -> bool:
    if not single_cmd:
        return single_cmd

    if run_command == "srun" and getenv("SLURM_HET_SIZE") is not None:
        msg = (
            "srun can not launch an orchestrator with single_cmd=True in "
            + "a hetereogeneous job. Automatically switching to single_cmd=False."
        )
        logger.info(msg)
        return False

    if not batch:
        return single_cmd

    if run_command == "aprun":
        msg = (
            "aprun can not launch an orchestrator with batch=True and "
            + "single_cmd=True. Automatically switching to single_cmd=False."
        )
        logger.info(msg)
        return False

    return single_cmd


def _check_local_constraints(launcher: str, batch: bool) -> None:
    """Check that the local launcher is not launched with invalid batch config"""
    if launcher == "local" and batch:
        msg = "Local orchestrator can not be launched with batch=True"
        raise SmartSimError(msg)


class Orchestrator(EntityList):
    """The Orchestrator is an in-memory database that can be launched
    alongside entities in SmartSim. Data can be transferred between
    entities by using one of the Python, C, C++ or Fortran clients
    within an entity.
    """

    def __init__(
        self,
        port: int = 6379,
        interface: t.Union[str, t.List[str]] = "lo",
        launcher: str = "local",
        run_command: str = "auto",
        db_nodes: int = 1,
        batch: bool = False,
        hosts: t.Optional[t.List[str]] = None,
        account: t.Optional[str] = None,
        time: t.Optional[str] = None,
        alloc: t.Optional[str] = None,
        single_cmd: bool = False,
        **kwargs: t.Any,
    ) -> None:
        """Initialize an Orchestrator reference for local launch

        :param port: TCP/IP port, defaults to 6379
        :type port: int, optional
        :param interface: network interface(s), defaults to "lo"
        :type interface: str, list[str], optional

        Extra configurations for RedisAI

        See https://oss.redislabs.com/redisai/configuration/

        :param threads_per_queue: threads per GPU device
        :type threads_per_queue: int, optional
        :param inter_op_threads: threads accross CPU operations
        :type inter_op_threads: int, optional
        :param intra_op_threads: threads per CPU operation
        :type intra_op_threads: int, optional
        """
        launcher, run_command = _autodetect(launcher, run_command)

        _check_run_command(launcher, run_command)
        _check_local_constraints(launcher, batch)

        single_cmd = _get_single_command(run_command, batch, single_cmd)

        self.launcher = launcher
        self.run_command = run_command

        self.ports: t.List[int] = []
        self.path = getcwd()
        self._hosts: t.List[str] = []
        if isinstance(interface, str):
            interface = [interface]
        self._interfaces = interface
        self._check_network_interface()
        self.queue_threads = kwargs.get("threads_per_queue", None)
        self.inter_threads = kwargs.get("inter_op_threads", None)
        self.intra_threads = kwargs.get("intra_op_threads", None)
        if self.launcher == "lsf":
            gpus_per_shard = kwargs.pop("gpus_per_shard", 0)
            cpus_per_shard = kwargs.pop("cpus_per_shard", 4)
        else:
            gpus_per_shard = None
            cpus_per_shard = None

        super().__init__(
            "orchestrator",
            self.path,
            port=port,
            interface=interface,
            db_nodes=db_nodes,
            batch=batch,
            launcher=launcher,
            run_command=run_command,
            alloc=alloc,
            single_cmd=single_cmd,
            gpus_per_shard=gpus_per_shard,
            cpus_per_shard=cpus_per_shard,
            **kwargs,
        )

        # detect if we can find at least the redis binaries. We
        # don't want to force the user to launch with RedisAI so
        # it's ok if that isn't present.
        try:
            # try to obtain redis binaries needed to launch Redis
            # will raise SSConfigError if not found
            self._redis_exe  # pylint: disable=W0104
            self._redis_conf  # pylint: disable=W0104
            CONFIG.database_cli  # pylint: disable=W0104
        except SSConfigError as e:
            msg = "SmartSim not installed with pre-built extensions (Redis)\n"
            msg += "Use the `smart` cli tool to install needed extensions\n"
            msg += "or set REDIS_PATH and REDIS_CLI_PATH in your environment\n"
            msg += "See documentation for more information"
            raise SSConfigError(msg) from e

        if launcher != "local":
            self.batch_settings = self._build_batch_settings(
                db_nodes,
                alloc or "",
                batch,
                account or "",
                time or "",
                launcher=launcher,
                **kwargs,
            )
            if hosts:
                self.set_hosts(hosts)
            elif not hosts and run_command == "mpirun":
                raise SmartSimError(
                    "hosts argument is required when launching Orchestrator with mpirun"
                )
            self._reserved_run_args: t.Dict[t.Type[RunSettings], t.List[str]] = {}
            self._reserved_batch_args: t.Dict[t.Type[BatchSettings], t.List[str]] = {}
            self._fill_reserved()

    @property
    def num_shards(self) -> int:
        """Return the number of DB shards contained in the orchestrator.
        This might differ from the number of ``DBNode`` objects, as each
        ``DBNode`` may start more than one shard (e.g. with MPMD).

        :returns: num_shards
        :rtype: int
        """
        return self.db_nodes

    @property
    def hosts(self) -> t.List[str]:
        """Return the hostnames of orchestrator instance hosts

        Note that this will only be populated after the orchestrator
        has been launched by SmartSim.

        :return: hostnames
        :rtype: list[str]
        """
        if not self._hosts:
            self._hosts = self._get_db_hosts()
        return self._hosts

    def remove_stale_files(self) -> None:
        """Can be used to remove database files of a previous launch"""

        for db in self.dbnodes:
            db.remove_stale_dbnode_files()

    def get_address(self) -> t.List[str]:
        """Return database addresses

        :return: addresses
        :rtype: list[str]

        :raises SmartSimError: If database address cannot be found or is not active
        """
        if not self._hosts:
            raise SmartSimError("Could not find database address")
        if not self.is_active():
            raise SmartSimError("Database is not active")
        return self._get_address()

    def _get_address(self) -> t.List[str]:
        addresses: t.List[str] = []
        for ip_address, port in itertools.product(self._hosts, self.ports):
            addresses.append(":".join((ip_address, str(port))))
        return addresses

    def is_active(self) -> bool:
        """Check if the database is active

        :return: True if database is active, False otherwise
        :rtype: bool
        """
        if not self._hosts:
            return False

        return db_is_active(self._hosts, self.ports, self.num_shards)

    @property
    def _rai_module(self) -> str:
        """Get the RedisAI module from third-party installations

        :return: path to module or "" if not found
        :rtype: str
        """
        module = ["--loadmodule", CONFIG.redisai]
        if self.queue_threads:
            module.append(f"THREADS_PER_QUEUE {self.queue_threads}")
        if self.inter_threads:
            module.append(f"INTER_OP_PARALLELISM {self.inter_threads}")
        if self.intra_threads:
            module.append(f"INTRA_OP_PARALLELISM {self.intra_threads}")
        return " ".join(module)

    @property
    def _redis_exe(self) -> str:
        return CONFIG.database_exe

    @property
    def _redis_conf(self) -> str:
        return CONFIG.database_conf

    def set_cpus(self, num_cpus: int) -> None:
        """Set the number of CPUs available to each database shard

        This effectively will determine how many cpus can be used for
        compute threads, background threads, and network I/O.

        :param num_cpus: number of cpus to set
        :type num_cpus: int
        """
        if self.batch:
            if self.launcher in ["pbs", "cobalt"]:
                if hasattr(self, "batch_settings") and self.batch_settings:
                    if hasattr(self.batch_settings, "set_ncpus"):
                        self.batch_settings.set_ncpus(num_cpus)
            if self.launcher == "slurm":
                if hasattr(self, "batch_settings") and self.batch_settings:
                    if hasattr(self.batch_settings, "set_cpus_per_task"):
                        self.batch_settings.set_cpus_per_task(num_cpus)

        for db in self.dbnodes:
            db.run_settings.set_cpus_per_task(num_cpus)
            if db.is_mpmd and hasattr(db.run_settings, "mpmd"):
                for mpmd in db.run_settings.mpmd:
                    mpmd.set_cpus_per_task(num_cpus)

    def set_walltime(self, walltime: str) -> None:
        """Set the batch walltime of the orchestrator

        Note: This will only effect orchestrators launched as a batch

        :param walltime: amount of time e.g. 10 hours is 10:00:00
        :type walltime: str
        :raises SmartSimError: if orchestrator isn't launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set walltime")

        if hasattr(self, "batch_settings") and self.batch_settings:
            self.batch_settings.set_walltime(walltime)

    def set_hosts(self, host_list: t.List[str]) -> None:
        """Specify the hosts for the ``Orchestrator`` to launch on

        :param host_list: list of host (compute node names)
        :type host_list: str, list[str]
        :raises TypeError: if wrong type
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        # TODO check length
        if self.batch:
            if hasattr(self, "batch_settings") and self.batch_settings:
                self.batch_settings.set_hostlist(host_list)

        if self.launcher == "lsf":
            for db in self.dbnodes:
                db.set_hosts(host_list)
        else:
            for host, db in zip(host_list, self.dbnodes):
                if isinstance(db.run_settings, AprunSettings):
                    if not self.batch:
                        db.run_settings.set_hostlist([host])
                else:
                    db.run_settings.set_hostlist([host])

                if db.is_mpmd and hasattr(db.run_settings, "mpmd"):
                    for i, mpmd_runsettings in enumerate(db.run_settings.mpmd):
                        mpmd_runsettings.set_hostlist(host_list[i + 1])

    def set_batch_arg(self, arg: str, value: t.Optional[str] = None) -> None:
        """Set a batch argument the orchestrator should launch with

        Some commonly used arguments such as --job-name are used
        by SmartSim and will not be allowed to be set.

        :param arg: batch argument to set e.g. "exclusive"
        :type arg: str
        :param value: batch param - set to None if no param value
        :type value: str | None
        :raises SmartSimError: if orchestrator not launching as batch
        """
        if not hasattr(self, "batch_settings") or not self.batch_settings:
            raise SmartSimError("Not running as batch, cannot set batch_arg")

        if arg in self._reserved_batch_args[type(self.batch_settings)]:
            logger.warning(
                f"Can not set batch argument {arg}: "
                "it is a reserved keyword in Orchestrator"
            )
        else:
            if hasattr(self, "batch_settings") and self.batch_settings:
                self.batch_settings.batch_args[arg] = value

    def set_run_arg(self, arg: str, value: t.Optional[str] = None) -> None:
        """Set a run argument the orchestrator should launch
        each node with (it will be passed to `jrun`)

        Some commonly used arguments are used
        by SmartSim and will not be allowed to be set.
        For example, "n", "N", etc.

        :param arg: run argument to set
        :type arg: str
        :param value: run parameter - set to None if no parameter value
        :type value: str | None
        """
        if arg in self._reserved_run_args[type(self.entities[0].run_settings)]:
            logger.warning(
                f"Can not set batch argument {arg}: "
                "it is a reserved keyword in Orchestrator"
            )
        else:
            for db in self.dbnodes:
                db.run_settings.run_args[arg] = value
                if db.is_mpmd and hasattr(db.run_settings, "mpmd"):
                    for mpmd in db.run_settings.mpmd:
                        mpmd.run_args[arg] = value

    def enable_checkpoints(self, frequency: int) -> None:
        """Sets the database's save configuration to save the
        DB every 'frequency' seconds given that at least one
        write operation against the DB occurred in that time.
        For example, if `frequency` is 900, then the database
        will save to disk after 900 seconds if there is at least
        1 change to the dataset.

        :param frequency: the given number of seconds before the DB saves
        :type frequency: int
        """
        self.set_db_conf("save", str(frequency) + " 1")

    def set_max_memory(self, mem: int) -> None:
        """Sets the max memory configuration. By default there is no memory limit.
        Setting max memory to zero also results in no memory limit. Once a limit is
        surpassed, keys will be removed according to the eviction strategy. The
        specified memory size is case insensitive and supports the typical forms of:
        1k => 1000 bytes
        1kb => 1024 bytes
        1m => 1000000 bytes
        1mb => 1024*1024 bytes
        1g => 1000000000 bytes
        1gb => 1024*1024*1024 bytes

        :param mem: the desired max memory size e.g. 3gb
        :type mem: str

        :raises SmartSimError: If 'mem' is an invalid memory value
        :raises SmartSimError: If database is not active
        """
        self.set_db_conf("maxmemory", mem)

    def set_eviction_strategy(self, strategy: str) -> None:
        """Sets how the database will select what to remove when
        'maxmemory' is reached. The default is noeviction.

        :param strategy: The max memory policy to use
            e.g. "volatile-lru", "allkeys-lru", etc.
        :type strategy: str

        :raises SmartSimError: If 'strategy' is an invalid maxmemory policy
        :raises SmartSimError: If database is not active
        """
        self.set_db_conf("maxmemory-policy", strategy)

    def set_max_clients(self, clients: int = 50_000) -> None:
        """Sets the max number of connected clients at the same time.
        When the number of DB shards contained in the orchestrator is
        more than two, then every node will use two connections, one
        incoming and another outgoing.

        :param clients: the maximum number of connected clients
        :type clients: int, optional
        """
        self.set_db_conf("maxclients", str(clients))

    def set_max_message_size(self, size: int = 1_073_741_824) -> None:
        """Sets the database's memory size limit for bulk requests,
        which are elements representing single strings. The default
        is 1 gigabyte. Message size must be greater than or equal to 1mb.
        The specified memory size should be an integer that represents
        the number of bytes. For example, to set the max message size
        to 1gb, use 1024*1024*1024.

        :param size: maximum message size in bytes
        :type size: int, optional
        """
        self.set_db_conf("proto-max-bulk-len", str(size))

    def set_db_conf(self, key: str, value: t.Union[int, str]) -> None:
        """Set any valid configuration at runtime without the need
        to restart the database. All configuration parameters
        that are set are immediately loaded by the database and
        will take effect starting with the next command executed.

        :param key: the configuration parameter
        :type key: str
        :param value: the database configuration parameter's new value
        :type value: str
        """
        if self.is_active():
            addresses = []
            for host in self.hosts:
                for port in self.ports:
                    address = ":".join([get_ip_from_host(host), str(port)])
                    addresses.append(address)

            is_cluster = self.num_shards > 2
            client = Client(address=addresses[0], cluster=is_cluster)
            try:
                for address in addresses:
                    client.config_set(key, value, address)

            except RedisReplyError:
                raise SmartSimError(
                    f"Invalid CONFIG key-value pair ({key}: {value})"
                ) from None
            except TypeError:
                raise TypeError(
                    "Incompatible function arguments. The key and value used for "
                    "setting the database configurations must be strings."
                ) from None
        else:
            raise SmartSimError(
                "The SmartSim Orchestrator must be active in order to set the "
                "database's configurations."
            )

    @staticmethod
    def _build_batch_settings(
        db_nodes: int,
        alloc: str,
        batch: bool,
        account: str,
        time: str,
        **kwargs: t.Any,
    ) -> t.Optional[BatchSettings]:
        batch_settings = None
        launcher = kwargs.pop("launcher")

        # enter this conditional if user has not specified an allocation to run
        # on or if user specified batch=False (alloc will be found through env)
        if not alloc and batch:
            batch_settings = create_batch_settings(
                launcher, nodes=db_nodes, time=time, account=account, **kwargs
            )

        return batch_settings

    def _build_run_settings(
        self, exe: str, exe_args: t.List[t.List[str]], **kwargs: t.Any
    ) -> RunSettings:
        run_args = kwargs.pop("run_args", {})
        db_nodes = kwargs.get("db_nodes", 1)
        single_cmd = kwargs.get("single_cmd", True)
        mpmd_nodes = single_cmd and db_nodes > 1

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

        # Put it back in case it is needed again
        kwargs["run_args"] = run_args

        return run_settings

    @staticmethod
    def _build_run_settings_lsf(
        exe: str, exe_args: t.List[t.List[str]], **kwargs: t.Any
    ) -> t.Optional[JsrunSettings]:
        run_args = kwargs.pop("run_args", {})
        cpus_per_shard = kwargs.get("cpus_per_shard", None)
        gpus_per_shard = kwargs.get("gpus_per_shard", None)
        erf_rs: t.Optional[JsrunSettings] = None

        # We always run the DB on cpus 0:cpus_per_shard-1
        # and gpus 0:gpus_per_shard-1
        for shard_id, args in enumerate(exe_args):
            host = shard_id
            run_args["launch_distribution"] = "packed"

            run_settings = JsrunSettings(exe, args, run_args=run_args.copy())
            run_settings.set_binding("none")

            # This makes sure output is written to orchestrator_0.out,
            # orchestrator_1.out, and so on
            run_settings.set_individual_output("_%t")

            erf_sets = {
                "rank": str(shard_id),
                "host": str(1 + host),
                "cpu": "{" + f"0:{cpus_per_shard}" + "}",
            }

            if gpus_per_shard > 1:  # pragma: no-cover
                erf_sets["gpu"] = "{" + f"0-{gpus_per_shard-1}" + "}"
            elif gpus_per_shard > 0:
                erf_sets["gpu"] = "{" + str(0) + "}"

            run_settings.set_erf_sets(erf_sets)

            if not erf_rs:
                erf_rs = run_settings
                continue

            erf_rs.make_mpmd(run_settings)

        kwargs["run_args"] = run_args

        return erf_rs

    def _initialize_entities(self, **kwargs: t.Any) -> None:
        self.db_nodes = int(kwargs.get("db_nodes", 1))
        single_cmd = kwargs.get("single_cmd", True)

        if int(self.db_nodes) == 2:
            raise SSUnsupportedError("Orchestrator does not support clusters of size 2")

        if self.launcher == "local" and self.db_nodes > 1:
            raise ValueError(
                "Local Orchestrator does not support multiple database shards"
            )

        mpmd_nodes = (single_cmd and self.db_nodes > 1) or self.launcher == "lsf"

        if mpmd_nodes:
            self._initialize_entities_mpmd(**kwargs)
        else:
            port = kwargs.get("port", 6379)
            cluster = not bool(self.db_nodes < 3)

            for db_id in range(self.db_nodes):
                db_node_name = "_".join((self.name, str(db_id)))

                # create the exe_args list for launching multiple databases
                # per node. also collect port range for dbnode
                start_script_args = self._get_start_script_args(
                    db_node_name, port, cluster
                )

                # if only launching 1 db per command, we don't need a
                # list of exe args lists
                run_settings = self._build_run_settings(
                    sys.executable, [start_script_args], **kwargs
                )

                node = DBNode(
                    db_node_name,
                    self.path,
                    run_settings,
                    [port],
                    [db_node_name + ".out"],
                )
                self.entities.append(node)

            self.ports = [port]

    def _initialize_entities_mpmd(self, **kwargs: t.Any) -> None:
        port = kwargs.get("port", 6379)
        cluster = not bool(self.db_nodes < 3)

        exe_args_mpmd: t.List[t.List[str]] = []

        for db_id in range(self.db_nodes):
            db_shard_name = "_".join((self.name, str(db_id)))
            # create the exe_args list for launching multiple databases
            # per node. also collect port range for dbnode
            start_script_args = self._get_start_script_args(
                db_shard_name, port, cluster
            )
            exe_args = " ".join(start_script_args)
            exe_args_mpmd.append(sh_split(exe_args))

        run_settings: t.Optional[RunSettings] = None

        if self.launcher == "lsf":
            run_settings = self._build_run_settings_lsf(
                sys.executable, exe_args_mpmd, **kwargs
            )
            output_files = [
                "_".join((self.name, str(db_id))) + ".out"
                for db_id in range(self.db_nodes)
            ]
        else:
            run_settings = self._build_run_settings(
                sys.executable, exe_args_mpmd, **kwargs
            )
            output_files = [self.name + ".out"]

        if not run_settings:
            raise ValueError(f"Could not build run settings for {self.launcher}")

        node = DBNode(self.name, self.path, run_settings, [port], output_files)
        node.is_mpmd = True
        node.num_shards = self.db_nodes
        self.entities.append(node)

        self.ports = [port]

    @staticmethod
    def _get_cluster_args(name: str, port: int) -> t.List[str]:
        """Create the arguments necessary for cluster creation"""
        cluster_conf = "".join(("nodes-", name, "-", str(port), ".conf"))
        db_args = ["--cluster-enabled yes", "--cluster-config-file", cluster_conf]
        return db_args

    def _get_start_script_args(
        self, name: str, port: int, cluster: bool
    ) -> t.List[str]:
        start_script_args = [
            "-m",
            "smartsim._core.entrypoints.redis",  # entrypoint
            "+ifname=" + ",".join(self._interfaces),  # pass interface to start script
            "+command",  # command flag for argparser
            self._redis_exe,  # redis-server
            self._redis_conf,  # redis.conf file
            self._rai_module,  # redisai.so
            "--port",  # redis port
            str(port),  # port number
        ]
        if cluster:
            start_script_args += self._get_cluster_args(name, port)

        return start_script_args

    @property
    def dbnodes(self) -> t.List[DBNode]:
        """
        Helper property to cast self.entities to DBNode type for type correctness
        """
        dbnodes = [node for node in self.entities if isinstance(node, DBNode)]
        return dbnodes

    def _get_db_hosts(self) -> t.List[str]:
        hosts = []
        for db in self.dbnodes:
            if not db.is_mpmd:
                hosts.append(db.host)
            else:
                hosts.extend(db.hosts)
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
        self._reserved_batch_args[CobaltBatchSettings] = [
            "cwd",
            "error",
            "e",
            "output",
            "o",
            "outputprefix",
            "N",
            "l",
            "jobname",
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
