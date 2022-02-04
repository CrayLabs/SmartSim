# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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
import os
from os import getcwd
from pathlib import Path
from shlex import split as sh_split

import psutil
import redis

from .._core.utils import check_cluster_status
from .._core.config import CONFIG
from .._core.utils.helpers import is_valid_cmd
from ..entity import DBNode, EntityList
from ..error import SmartSimError, SSConfigError, SSInternalError, SSUnsupportedError
from ..log import get_logger
from ..settings import (
    AprunSettings,
    BsubBatchSettings,
    CobaltBatchSettings,
    JsrunSettings,
    MpirunSettings,
    QsubBatchSettings,
    SbatchSettings,
    SrunSettings,
)
from ..settings.settings import create_batch_settings, create_run_settings
from ..wlm import detect_launcher

logger = get_logger(__name__)


class Orchestrator(EntityList):
    """The Orchestrator is an in-memory database that can be launched
    alongside entities in SmartSim. Data can be transferred between
    entities by using one of the Python, C, C++ or Fortran clients
    within an entity.
    """

    def __init__(
        self,
        port=6379,
        interface="lo",
        launcher="local",
        run_command="auto",
        db_nodes=1,
        batch=False,
        hosts=None,
        account=None,
        time=None,
        alloc=None,
        single_cmd=False,
        **kwargs,
    ):
        """Initialize an Orchestrator reference for local launch

        :param port: TCP/IP port, defaults to 6379
        :type port: int, optional
        :param interface: network interface, defaults to "lo"
        :type interface: str, optional

        Extra configurations for RedisAI

        See https://oss.redislabs.com/redisai/configuration/

        :param threads_per_queue: threads per GPU device
        :type threads_per_queue: int, optional
        :param inter_op_threads: threads accross CPU operations
        :type inter_op_threads: int, optional
        :param intra_op_threads: threads per CPU operation
        :type intra_op_threads: int, optional
        """

        if launcher == "auto":
            launcher = detect_launcher()

        by_launcher = {
            "slurm": ["srun", "mpirun"],
            "pbs": ["aprun", "mpirun"],
            "cobalt": ["aprun", "mpirun"],
            "lsf": ["jsrun"],
            "local": [None],
        }

        def _detect_command(launcher):
            if launcher in by_launcher:
                for cmd in by_launcher[launcher]:
                    if launcher == "local":
                        return cmd
                    if is_valid_cmd(cmd):
                        return cmd
            msg = f"Could not automatically detect a run command to use for launcher {launcher}"
            msg += f"\nSearched for and could not find the following commands: {by_launcher[launcher]}"
            raise SmartSimError(msg)

        if run_command == "auto":
            run_command = _detect_command(launcher)

        if run_command not in by_launcher[launcher]:
            msg = f"Run command {run_command} is not supported on launcher {launcher}\n"
            msg += f"Supported run commands for the given launcher are: {by_launcher[launcher]}"
            raise SmartSimError(msg)

        if launcher == "local" and batch:
            msg = "Local launcher can not be launched with batch=True"
            raise SmartSimError(msg)

        self.launcher = launcher
        self.run_command = run_command

        self.ports = []
        self.path = getcwd()
        self._hosts = []
        self._interface = interface
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
            self._redis_exe
            self._redis_conf
            CONFIG.redis_cli
        except SSConfigError as e:
            msg = "SmartSim not installed with pre-built extensions (Redis)\n"
            msg += "Use the `smart` cli tool to install needed extensions\n"
            msg += "or set REDIS_PATH and REDIS_CLI_PATH in your environment\n"
            msg += "See documentation for more information"
            raise SSConfigError(msg) from e

        if launcher != "local":
            self.batch_settings = self._build_batch_settings(
                db_nodes, alloc, batch, account, time, launcher=launcher, **kwargs
            )
            if hosts:
                self.set_hosts(hosts)
            elif not hosts and run_command == "mpirun":
                raise SmartSimError(
                    "hosts argument is required when launching Orchestrator with mpirun"
                )
            self._reserved_run_args = {}
            self._reserved_batch_args = {}
            self._fill_reserved()

    @property
    def num_shards(self):
        """Return the number of DB shards contained in the orchestrator.
        This might differ from the number of ``DBNode`` objects, as each
        ``DBNode`` may start more than one shard (e.g. with MPMD).

        :returns: num_shards
        :rtype: int
        """
        return self.db_nodes

    @property
    def hosts(self):
        """Return the hostnames of orchestrator instance hosts

        Note that this will only be populated after the orchestrator
        has been launched by SmartSim.

        :return: hostnames
        :rtype: list[str]
        """
        if not self._hosts:
            self._hosts = self._get_db_hosts()
        return self._hosts

    def remove_stale_files(self):
        """Can be used to remove database files of a previous launch"""

        for dbnode in self.entities:
            dbnode.remove_stale_dbnode_files()

    def get_address(self):
        """Return database addresses

        :return: addresses
        :rtype: list[str]

        :raises SmartSimError: If database address cannot be found or is not active
        """
        if not self._hosts:
            raise SmartSimError("Could not find database address")
        elif not self.is_active():
            raise SmartSimError("Database is not active")
        return self._get_address()

    def _get_address(self):
        addresses = []
        for ip, port in itertools.product(self._hosts, self.ports):
            addresses.append(":".join((ip, str(port))))
        return addresses

    def is_active(self):
        """Check if the database is active

        :return: True if database is active, False otherwise
        :rtype: bool
        """
        if not self._hosts:
            return False

        # if single shard
        if self.num_shards < 2:
            host = self._hosts[0]
            port = self.ports[0]
            try:
                client = redis.Redis(host=host, port=port, db=0)
                if client.ping():
                    return True
                return False
            except redis.RedisError:
                return False
        # if a cluster
        else:
            try:
                check_cluster_status(self._hosts, self.ports, trials=1)
                return True
            # we expect this to fail if the cluster is not active
            except SSInternalError:
                return False

    @property
    def _rai_module(self):
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
    def _redis_launch_script(self):
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        script_path = current_dir.joinpath("redis_starter.py").resolve()
        return str(script_path)

    @property
    def _redis_exe(self):
        return CONFIG.redis_exe

    @property
    def _redis_conf(self):
        return CONFIG.redis_conf

    def set_cpus(self, num_cpus):
        """Set the number of CPUs available to each database shard

        This effectively will determine how many cpus can be used for
        compute threads, background threads, and network I/O.

        :param num_cpus: number of cpus to set
        :type num_cpus: int
        """
        if self.batch:
            if self.launcher == "pbs" or self.launcher == "cobalt":
                self.batch_settings.set_ncpus(num_cpus)
            if self.launcher == "slurm":
                self.batch_settings.set_cpus_per_task(num_cpus)
        for db in self:
            db.run_settings.set_cpus_per_task(num_cpus)
            if db._mpmd:
                for mpmd in db.run_settings.mpmd:
                    mpmd.set_cpus_per_task(num_cpus)

    def set_walltime(self, walltime):
        """Set the batch walltime of the orchestrator

        Note: This will only effect orchestrators launched as a batch

        :param walltime: amount of time e.g. 10 hours is 10:00:00
        :type walltime: str
        :raises SmartSimError: if orchestrator isn't launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set walltime")
        self.batch_settings.set_walltime(walltime)

    def set_hosts(self, host_list):
        """Specify the hosts for the ``Orchestrator`` to launch on

        :param host_list: list of host (compute node names)
        :type host_list: str, list[str]
        :raises TypeError: if wrong type
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all([isinstance(host, str) for host in host_list]):
            raise TypeError("host_list argument must be list of strings")
        # TODO check length
        if self.batch:
            self.batch_settings.set_hostlist(host_list)

        if self.launcher == "lsf":
            for db in self:
                db.set_hosts(host_list)
        else:
            for host, db in zip(host_list, self.entities):
                if isinstance(db.run_settings, AprunSettings):
                    if not self.batch:
                        db.run_settings.set_hostlist([host])
                else:
                    db.run_settings.set_hostlist([host])
                if db._mpmd:
                    for i, mpmd_runsettings in enumerate(db.run_settings.mpmd):
                        mpmd_runsettings.set_hostlist(host_list[i + 1])

    def set_batch_arg(self, arg, value):
        """Set a batch argument the orchestrator should launch with

        Some commonly used arguments such as --job-name are used
        by SmartSim and will not be allowed to be set.

        :param arg: batch argument to set e.g. "exclusive"
        :type arg: str
        :param value: batch param - set to None if no param value
        :type value: str | None
        :raises SmartSimError: if orchestrator not launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set batch_arg")
        if arg in self._reserved_batch_args[type(self.batch_settings)]:
            logger.warning(
                f"Can not set batch argument {arg}: it is a reserved keyword in Orchestrator"
            )
        else:
            self.batch_settings.batch_args[arg] = value

    def set_run_arg(self, arg, value):
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
                f"Can not set run argument {arg}: it is a reserved keyword in Orchestrator"
            )
        else:
            for db in self.entities:
                db.run_settings.run_args[arg] = value
                if db._mpmd:
                    for mpmd in db.run_settings.mpmd:
                        mpmd.run_args[arg] = value

    def _build_batch_settings(self, db_nodes, alloc, batch, account, time, **kwargs):
        batch_settings = None
        launcher = kwargs.pop("launcher")

        # enter this conditional if user has not specified an allocation to run
        # on or if user specified batch=False (alloc will be found through env)
        if not alloc and batch:
            batch_settings = create_batch_settings(
                launcher, nodes=db_nodes, time=time, account=account, **kwargs
            )

        return batch_settings

    def _build_run_settings(self, exe, exe_args, **kwargs):
        run_command = kwargs.get("run_command")
        run_args = kwargs.get("run_args", {})
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
                exe=exe, exe_args=exe_args, run_args=run_args, **kwargs
            )

            if self.launcher != "local":
                run_settings.set_tasks(1)

        if self.launcher != "local":
            run_settings.set_tasks_per_node(1)

        if isinstance(run_settings, SrunSettings):
            run_args["nodes"] = 1 if not mpmd_nodes else db_nodes

        return run_settings

    def _build_run_settings_lsf(self, exe, exe_args, **kwargs):
        run_args = kwargs.get("run_args", {}).copy()
        cpus_per_shard = kwargs.get("cpus_per_shard", None)
        gpus_per_shard = kwargs.get("gpus_per_shard", None)
        old_host = None
        erf_rs = None
        for shard_id, args in enumerate(exe_args):
            host = shard_id
            run_args["launch_distribution"] = "packed"

            run_settings = JsrunSettings(exe, args, run_args=run_args)
            run_settings.set_binding("none")

            # This makes sure output is written to orchestrator_0.out, orchestrator_1.out, and so on
            run_settings.set_individual_output("_%t")
            # tell step to create a mpmd executable even if we only have one task
            # because we need to specify the host
            if host != old_host:
                assigned_smts = 0
                assigned_gpus = 0
            old_host = host

            erf_sets = {
                "rank": str(shard_id),
                "host": str(1 + host),
                "cpu": "{" + f"{assigned_smts}:{cpus_per_shard}" + "}",
            }

            assigned_smts += cpus_per_shard
            if gpus_per_shard > 1:  # pragma: no-cover
                erf_sets["gpu"] = (
                    "{" + f"{assigned_gpus}-{assigned_gpus+self.gpus_per_shard-1}" + "}"
                )
            elif gpus_per_shard > 0:
                erf_sets["gpu"] = "{" + f"{assigned_gpus}" + "}"
            assigned_gpus += gpus_per_shard

            run_settings.set_erf_sets(erf_sets)

            if erf_rs:
                erf_rs.make_mpmd(run_settings)
            else:
                run_settings.make_mpmd()
                erf_rs = run_settings

        return erf_rs

    def _initialize_entities(self, **kwargs):
        port = kwargs.get("port", 6379)
        self.db_nodes = kwargs.get("db_nodes", 1)
        single_cmd = kwargs.get("single_cmd", True)

        mpmd_nodes = (single_cmd and self.db_nodes > 1) or self.launcher == "lsf"

        cluster = not bool(self.db_nodes < 3)
        if int(self.db_nodes) == 2:
            raise SSUnsupportedError("Orchestrator does not support clusters of size 2")

        if self.launcher == "local" and self.db_nodes > 1:
            raise ValueError(
                "Local Orchestrator does not support multiple database shards"
            )

        if mpmd_nodes:
            exe_args_mpmd = []

        for db_id in range(self.db_nodes):
            db_shard_name = "_".join((self.name, str(db_id)))
            if not mpmd_nodes:
                db_node_name = db_shard_name
            else:
                db_node_name = self.name

            # create the exe_args list for launching multiple databases
            # per node. also collect port range for dbnode
            start_script_args = [
                self._redis_launch_script,  # redis_starter.py
                f"+ifname={self._interface}",  # pass interface to start script
                "+command",  # command flag for argparser
                self._redis_exe,  # redis-server
                self._redis_conf,  # redis6.conf file
                self._rai_module,  # redisai.so
                "--port",  # redis port
                str(port),  # port number
            ]
            if cluster:
                start_script_args += self._get_cluster_args(db_shard_name, port)

            exe_args = " ".join(start_script_args)

            if not mpmd_nodes:
                # if only launching 1 db_per_host, we don't need a list of exe args lists
                run_settings = self._build_run_settings("python", exe_args, **kwargs)

                node = DBNode(db_node_name, self.path, run_settings, [port])
                self.entities.append(node)
            else:
                exe_args_mpmd.append(sh_split(exe_args))

        if mpmd_nodes:
            if self.launcher == "lsf":
                run_settings = self._build_run_settings_lsf(
                    "python", exe_args_mpmd, **kwargs
                )
            else:
                run_settings = self._build_run_settings(
                    "python", exe_args_mpmd, **kwargs
                )
            node = DBNode(db_node_name, self.path, run_settings, [port])
            node._mpmd = True
            node._shard_ids = range(self.db_nodes)
            self.entities.append(node)

        self.ports = [port]

    @staticmethod
    def _get_cluster_args(name, port):
        """Create the arguments necessary for cluster creation"""
        cluster_conf = "".join(("nodes-", name, "-", str(port), ".conf"))
        db_args = ["--cluster-enabled yes", "--cluster-config-file", cluster_conf]
        return db_args

    def _get_db_hosts(self):
        hosts = []
        for dbnode in self.entities:
            if not dbnode._mpmd:
                hosts.append(dbnode.host)
            else:
                hosts.extend(dbnode.hosts)
        return hosts

    def _check_network_interface(self):
        net_if_addrs = psutil.net_if_addrs()
        if self._interface not in net_if_addrs and self._interface != "lo":
            available = list(net_if_addrs.keys())
            logger.warning(
                f"{self._interface} is not a valid network interface on this node. \n"
                "This could be because the head node doesn't have the same networks, if so, ignore this."
            )
            logger.warning(f"Found network interfaces are: {available}")

    def _fill_reserved(self):
        """Fill the reserved batch and run arguments dictionaries"""
        self._reserved_run_args[MpirunSettings] = [
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
