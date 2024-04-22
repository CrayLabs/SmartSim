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
import itertools
import sys
import typing as t
from os import environ, getcwd, getenv
from shlex import split as sh_split

import psutil
from smartredis import Client, ConfigOptions
from smartredis.error import RedisReplyError

from .._core.config import CONFIG
from .._core.utils import db_is_active
from .._core.utils.helpers import is_valid_cmd, unpack_db_identifier
from .._core.utils.network import get_ip_from_host
from ..entity import DBNode, EntityList, TelemetryConfiguration
from ..error import SmartSimError, SSConfigError, SSUnsupportedError
from ..log import get_logger
from ..servertype import CLUSTERED, STANDALONE
from ..settings import (
    AprunSettings,
    BsubBatchSettings,
    JsrunSettings,
    MpiexecSettings,
    MpirunSettings,
    OrterunSettings,
    PalsMpiexecSettings,
    QsubBatchSettings,
    SbatchSettings,
    SrunSettings,
)
from ..settings.base import BatchSettings, RunSettings
from ..settings.settings import create_batch_settings, create_run_settings
from ..wlm import detect_launcher

logger = get_logger(__name__)

by_launcher: t.Dict[str, t.List[str]] = {
    "slurm": ["srun", "mpirun", "mpiexec"],
    "pbs": ["aprun", "mpirun", "mpiexec"],
    "pals": ["mpiexec"],
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


class Orchestrator(EntityList[DBNode]):
    """The Orchestrator is an in-memory database that can be launched
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
        db_nodes: int = 1,
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
        db_identifier: str = "orchestrator",
        **kwargs: t.Any,
    ) -> None:
        """Initialize an ``Orchestrator`` reference for local launch

        Extra configurations for RedisAI

        See https://oss.redis.com/redisai/configuration/

        :param path: path to location of ``Orchestrator`` directory
        :param port: TCP/IP port
        :param interface: network interface(s)
        :param launcher: type of launcher being used, options are "slurm", "pbs",
                         "lsf", or "local". If set to "auto",
                         an attempt will be made to find an available launcher
                         on the system.
        :param run_command: specify launch binary or detect automatically
        :param db_nodes: number of database shards
        :param batch: run as a batch workload
        :param hosts: specify hosts to launch on
        :param account: account to run batch on
        :param time: walltime for batch 'HH:MM:SS' format
        :param alloc: allocation to launch database on
        :param single_cmd: run all shards with one (MPMD) command
        :param threads_per_queue: threads per GPU device
        :param inter_op_threads: threads across CPU operations
        :param intra_op_threads: threads per CPU operation
        :param db_identifier: an identifier to distinguish this orchestrator in
            multiple-database experiments
        """
        self.launcher, self.run_command = _autodetect(launcher, run_command)
        _check_run_command(self.launcher, self.run_command)
        _check_local_constraints(self.launcher, batch)
        single_cmd = _get_single_command(self.run_command, batch, single_cmd)
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
            name=db_identifier,
            path=str(path),
            port=port,
            interface=interface,
            db_nodes=db_nodes,
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
            raise SSConfigError(
                "SmartSim not installed with pre-built extensions (Redis)\n"
                "Use the `smart` cli tool to install needed extensions\n"
                "or set REDIS_PATH and REDIS_CLI_PATH in your environment\n"
                "See documentation for more information"
            ) from e

        if self.launcher != "local":
            self.batch_settings = self._build_batch_settings(
                db_nodes,
                alloc or "",
                batch,
                account or "",
                time or "",
                launcher=self.launcher,
                **kwargs,
            )
            if hosts:
                self.set_hosts(hosts)
            elif not hosts and self.run_command == "mpirun":
                raise SmartSimError(
                    "hosts argument is required when launching Orchestrator with mpirun"
                )
            self._reserved_run_args: t.Dict[t.Type[RunSettings], t.List[str]] = {}
            self._reserved_batch_args: t.Dict[t.Type[BatchSettings], t.List[str]] = {}
            self._fill_reserved()

    @property
    def db_identifier(self) -> str:
        """Return the DB identifier, which is common to a DB and all of its nodes

        :return: DB identifier
        """
        return self.name

    @property
    def num_shards(self) -> int:
        """Return the number of DB shards contained in the Orchestrator.
        This might differ from the number of ``DBNode`` objects, as each
        ``DBNode`` may start more than one shard (e.g. with MPMD).

        :returns: the number of DB shards contained in the Orchestrator
        """
        return sum(node.num_shards for node in self.entities)

    @property
    def db_nodes(self) -> int:
        """Read only property for the number of nodes an ``Orchestrator`` is
        launched across. Notice that SmartSim currently assumes that each shard
        will be launched on its own node. Therefore this property is currently
        an alias to the ``num_shards`` attribute.

        :returns: Number of database nodes
        """
        return self.num_shards

    @property
    def hosts(self) -> t.List[str]:
        """Return the hostnames of Orchestrator instance hosts

        Note that this will only be populated after the orchestrator
        has been launched by SmartSim.

        :return: the hostnames of Orchestrator instance hosts
        """
        if not self._hosts:
            self._hosts = self._get_db_hosts()
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
        """Can be used to remove database files of a previous launch"""

        for db in self.entities:
            db.remove_stale_dbnode_files()

    def get_address(self) -> t.List[str]:
        """Return database addresses

        :return: addresses

        :raises SmartSimError: If database address cannot be found or is not active
        """
        if not self._hosts:
            raise SmartSimError("Could not find database address")
        if not self.is_active():
            raise SmartSimError("Database is not active")
        return self._get_address()

    def _get_address(self) -> t.List[str]:
        return [
            f"{host}:{port}"
            for host, port in itertools.product(self._hosts, self.ports)
        ]

    def is_active(self) -> bool:
        """Check if the database is active

        :return: True if database is active, False otherwise
        """
        if not self._hosts:
            return False

        return db_is_active(self._hosts, self.ports, self.num_shards)

    @property
    def _rai_module(self) -> t.Tuple[str, ...]:
        """Get the RedisAI module from third-party installations

        :return: Tuple of args to pass to the orchestrator exe
                 to load and configure the RedisAI
        """
        module = ["--loadmodule", CONFIG.redisai]
        if self.queue_threads:
            module.extend(("THREADS_PER_QUEUE", str(self.queue_threads)))
        if self.inter_threads:
            module.extend(("INTER_OP_PARALLELISM", str(self.inter_threads)))
        if self.intra_threads:
            module.extend(("INTRA_OP_PARALLELISM", str(self.intra_threads)))
        return tuple(module)

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

        for db in self.entities:
            db.run_settings.set_cpus_per_task(num_cpus)
            if db.is_mpmd and hasattr(db.run_settings, "mpmd"):
                for mpmd in db.run_settings.mpmd:
                    mpmd.set_cpus_per_task(num_cpus)

    def set_walltime(self, walltime: str) -> None:
        """Set the batch walltime of the orchestrator

        Note: This will only effect orchestrators launched as a batch

        :param walltime: amount of time e.g. 10 hours is 10:00:00
        :raises SmartSimError: if orchestrator isn't launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set walltime")

        if hasattr(self, "batch_settings") and self.batch_settings:
            self.batch_settings.set_walltime(walltime)

    def set_hosts(self, host_list: t.Union[t.List[str], str]) -> None:
        """Specify the hosts for the ``Orchestrator`` to launch on

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
        if self.batch:
            if hasattr(self, "batch_settings") and self.batch_settings:
                self.batch_settings.set_hostlist(host_list)

        if self.launcher == "lsf":
            for db in self.entities:
                db.set_hosts(host_list)
        elif (
            self.launcher == "pals"
            and isinstance(self.entities[0].run_settings, PalsMpiexecSettings)
            and self.entities[0].is_mpmd
        ):
            # In this case, --hosts is a global option, set it to first run command
            self.entities[0].run_settings.set_hostlist(host_list)
        else:
            for host, db in zip(host_list, self.entities):
                if isinstance(db.run_settings, AprunSettings):
                    if not self.batch:
                        db.run_settings.set_hostlist([host])
                else:
                    db.run_settings.set_hostlist([host])

                if db.is_mpmd and hasattr(db.run_settings, "mpmd"):
                    for i, mpmd_runsettings in enumerate(db.run_settings.mpmd, 1):
                        mpmd_runsettings.set_hostlist(host_list[i])

    def set_batch_arg(self, arg: str, value: t.Optional[str] = None) -> None:
        """Set a batch argument the orchestrator should launch with

        Some commonly used arguments such as --job-name are used
        by SmartSim and will not be allowed to be set.

        :param arg: batch argument to set e.g. "exclusive"
        :param value: batch param - set to None if no param value
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
            self.batch_settings.batch_args[arg] = value

    def set_run_arg(self, arg: str, value: t.Optional[str] = None) -> None:
        """Set a run argument the orchestrator should launch
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
                "it is a reserved keyword in Orchestrator"
            )
        else:
            for db in self.entities:
                db.run_settings.run_args[arg] = value
                if db.is_mpmd and hasattr(db.run_settings, "mpmd"):
                    for mpmd in db.run_settings.mpmd:
                        mpmd.run_args[arg] = value

    def enable_checkpoints(self, frequency: int) -> None:
        """Sets the database's save configuration to save the DB every 'frequency'
        seconds given that at least one write operation against the DB occurred in
        that time. E.g., if `frequency` is 900, then the database will save to disk
        after 900 seconds if there is at least 1 change to the dataset.

        :param frequency: the given number of seconds before the DB saves
        """
        self.set_db_conf("save", f"{frequency} 1")

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
        :raises SmartSimError: If database is not active
        """
        self.set_db_conf("maxmemory", mem)

    def set_eviction_strategy(self, strategy: str) -> None:
        """Sets how the database will select what to remove when
        'maxmemory' is reached. The default is noeviction.

        :param strategy: The max memory policy to use
            e.g. "volatile-lru", "allkeys-lru", etc.
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
        """
        self.set_db_conf("proto-max-bulk-len", str(size))

    def set_db_conf(self, key: str, value: str) -> None:
        """Set any valid configuration at runtime without the need
        to restart the database. All configuration parameters
        that are set are immediately loaded by the database and
        will take effect starting with the next command executed.

        :param key: the configuration parameter
        :param value: the database configuration parameter's new value
        """
        if self.is_active():
            addresses = []
            for host in self.hosts:
                for port in self.ports:
                    addresses.append(":".join([get_ip_from_host(host), str(port)]))

            db_name, name = unpack_db_identifier(self.db_identifier, "_")

            environ[f"SSDB{db_name}"] = addresses[0]

            db_type = CLUSTERED if self.num_shards > 2 else STANDALONE
            environ[f"SR_DB_TYPE{db_name}"] = db_type

            options = ConfigOptions.create_from_environment(name)
            client = Client(options)

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
                launcher, nodes=db_nodes, time=time, account=account, **kwargs
            )

        return batch_settings

    def _build_run_settings(
        self,
        exe: str,
        exe_args: t.List[t.List[str]],
        *,
        run_args: t.Optional[t.Dict[str, t.Any]] = None,
        db_nodes: int = 1,
        single_cmd: bool = True,
        **kwargs: t.Any,
    ) -> RunSettings:
        run_args = {} if run_args is None else run_args
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
        db_nodes: int = 1,
        single_cmd: bool = True,
        port: int = 6379,
        **kwargs: t.Any,
    ) -> None:
        db_nodes = int(db_nodes)
        if db_nodes == 2:
            raise SSUnsupportedError("Orchestrator does not support clusters of size 2")

        if self.launcher == "local" and db_nodes > 1:
            raise ValueError(
                "Local Orchestrator does not support multiple database shards"
            )

        mpmd_nodes = (single_cmd and db_nodes > 1) or self.launcher == "lsf"

        if mpmd_nodes:
            self._initialize_entities_mpmd(
                db_nodes=db_nodes, single_cmd=single_cmd, port=port, **kwargs
            )
        else:
            cluster = db_nodes >= 3

            for db_id in range(db_nodes):
                db_node_name = "_".join((self.name, str(db_id)))

                # create the exe_args list for launching multiple databases
                # per node. also collect port range for dbnode
                start_script_args = self._get_start_script_args(
                    db_node_name, port, cluster
                )
                # if only launching 1 db per command, we don't need a
                # list of exe args lists
                run_settings = self._build_run_settings(
                    sys.executable, [start_script_args], port=port, **kwargs
                )

                node = DBNode(
                    db_node_name,
                    self.path,
                    run_settings,
                    [port],
                    [db_node_name + ".out"],
                    self.db_identifier,
                )
                self.entities.append(node)

            self.ports = [port]

    def _initialize_entities_mpmd(
        self, *, db_nodes: int = 1, port: int = 6379, **kwargs: t.Any
    ) -> None:
        cluster = db_nodes >= 3
        mpmd_node_name = self.name + "_0"
        exe_args_mpmd: t.List[t.List[str]] = []

        for db_id in range(db_nodes):
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
                sys.executable, exe_args_mpmd, db_nodes=db_nodes, port=port, **kwargs
            )
            output_files = [f"{self.name}_{db_id}.out" for db_id in range(db_nodes)]
        else:
            run_settings = self._build_run_settings(
                sys.executable, exe_args_mpmd, db_nodes=db_nodes, port=port, **kwargs
            )
            output_files = [mpmd_node_name + ".out"]
        if not run_settings:
            raise ValueError(f"Could not build run settings for {self.launcher}")
        node = DBNode(
            mpmd_node_name,
            self.path,
            run_settings,
            [port],
            output_files,
            db_identifier=self.db_identifier,
        )
        self.entities.append(node)
        self.ports = [port]

    def _get_start_script_args(
        self, name: str, port: int, cluster: bool
    ) -> t.List[str]:
        cmd = [
            "-m",
            "smartsim._core.entrypoints.redis",  # entrypoint
            f"+orc-exe={self._redis_exe}",  # redis-server
            f"+conf-file={self._redis_conf}",  # redis.conf file
            "+rai-module",  # load redisai.so
            *self._rai_module,
            f"+name={name}",  # name of node
            f"+port={port}",  # redis port
            f"+ifname={','.join(self._interfaces)}",  # pass interface to start script
        ]
        if cluster:
            cmd.append("+cluster")  # is the shard part of a cluster
        return cmd

    def _get_db_hosts(self) -> t.List[str]:
        hosts = []
        for db in self.entities:
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
