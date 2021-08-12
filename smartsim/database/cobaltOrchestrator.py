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

from ..config import CONFIG
from ..entity import DBNode
from ..error import SmartSimError, SSUnsupportedError
from ..settings import AprunSettings, CobaltBatchSettings, MpirunSettings
from ..utils import get_logger
from .orchestrator import Orchestrator

logger = get_logger(__name__)


class CobaltOrchestrator(Orchestrator):
    def __init__(
        self,
        port=6379,
        db_nodes=1,
        batch=True,
        hosts=None,
        run_command="aprun",
        interface="ipogif0",
        account=None,
        queue=None,
        time=None,
        **kwargs,
    ):
        """Initialize an Orchestrator reference for Cobalt based systems

        The orchestrator launches as a batch by default. If batch=False,
        at launch, the orchestrator will look for an interactive
        allocation to launch on.

        The Cobalt orchestrator does not support multiple databases per node.

        :param port: TCP/IP port, defaults to 6379
        :type port: int
        :param db_nodes: number of database shards, defaults to 1
        :type db_nodes: int, optional
        :param batch: Run as a batch workload, defaults to True
        :type batch: bool, optional
        :param hosts: specify hosts to launch on, defaults to None. Optional if not launching with OpenMPI
        :type hosts: list[str]
        :param run_command: specify launch binary. Options are ``mpirun`` and ``aprun``, defaults to "aprun".
        :type run_command: str, optional
        :param interface: network interface to use, defaults to "ipogif0"
        :type interface: str, optional
        :param account: account to run batch on
        :type account: str, optional
        :param queue: queue to launch batch in
        :type queue: str, optional
        :param time: walltime for batch 'HH:MM:SS' format
        :type time: str, optional
        """
        super().__init__(
            port,
            interface,
            db_nodes=db_nodes,
            batch=batch,
            run_command=run_command,
            **kwargs,
        )
        self.batch_settings = self._build_batch_settings(
            db_nodes, batch, account, queue, time
        )
        if hosts:
            self.set_hosts(hosts)
        elif not hosts and run_command == "mpirun":
            raise SmartSimError(
                "hosts argument is required when launching CobaltOrchestrator with OpenMPI"
            )
        self._reserved_run_args = {}
        self._reserved_batch_args = {}
        self._fill_reserved()

    def set_cpus(self, num_cpus):
        """Set the number of CPUs available to each database shard

        This effectively will determine how many cpus can be used for
        compute threads, background threads, and network I/O.

        :param num_cpus: number of cpus to set
        :type num_cpus: int
        """
        for db in self.entities:
            # Supported by MpirunSettings and AprunSettings
            db.run_settings.set_cpus_per_task(num_cpus)

    def set_walltime(self, walltime):
        """Set the batch walltime of the orchestrator

        Note: This will only effect orchestrators launched as a batch

        :param walltime: amount of time e.g. 10 hours is 10:00:00
        :type walltime: str
        :raises SmartSimError: if orchestrator isn't launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running in batch, cannot set walltime")
        self.batch_settings.set_walltime(walltime)

    def set_hosts(self, host_list):
        """Specify the hosts for the ``CobaltOrchestrator`` to launch on

        :param host_list: list of hosts (compute node names)
        :type host_list: str | list[str]
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
        for host, db in zip(host_list, self.entities):
            db.set_host(host)

            # Aprun doesn't like settings hosts in batch launch
            if isinstance(db.run_settings, AprunSettings):
                if not self.batch:
                    db.run_settings.set_hostlist([host])
            else:
                db.run_settings.set_hostlist([host])

    def set_batch_arg(self, arg, value):
        """Set a cobalt ``qsub`` argument

        Some commonly used arguments are used
        by SmartSim and will not be allowed to be set.
        For example, "cwd", "jobname", etc.

        :param arg: batch argument to set e.g. "exclusive"
        :type arg: str
        :param value: batch param - set to None if no param value
        :type value: str | None
        :raises SmartSimError: if orchestrator not launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set batch_arg")
        if arg in self._reserved_batch_args:
            logger.warning(
                f"Can not set batch argument {arg}: it is a reserved keyword in CobaltOrchestrator"
            )
        else:
            self.batch_settings.batch_args[arg] = value

    def set_run_arg(self, arg, value):
        """Set a run argument the orchestrator should launch
        each node with (it will be passed to `aprun`)

        Some commonly used arguments are used
        by SmartSim and will not be allowed to be set.
        For example, "wdir", "n", etc.

        :param arg: run argument to set
        :type arg: str
        :param value: run parameter - set to None if no parameter value
        :type value: str | None
        """
        if arg in self._reserved_run_args[type(self.entities[0].run_settings)]:
            logger.warning(
                f"Can not set run argument {arg}: it is a reserved keyword in CobaltOrchestrator"
            )
        else:
            for db in self.entities:
                db.run_settings.run_args[arg] = value

    def _build_run_settings(self, exe, exe_args, **kwargs):
        run_command = kwargs.get("run_command", "aprun")
        if run_command == "aprun":
            return self._build_aprun_settings(exe, exe_args, **kwargs)
        if run_command == "mpirun":
            return self._build_mpirun_settings(exe, exe_args, **kwargs)
        raise SSUnsupportedError(
            f"CobaltOrchestrator does not support {run_command} as a launch binary"
        )

    def _build_aprun_settings(self, exe, exe_args, **kwargs):
        run_args = kwargs.get("run_args", {})
        run_settings = AprunSettings(exe, exe_args, run_args=run_args)
        run_settings.set_tasks(1)
        run_settings.set_tasks_per_node(1)
        return run_settings

    def _build_mpirun_settings(self, exe, exe_args, **kwargs):
        run_args = kwargs.get("run_args", {})
        run_settings = MpirunSettings(exe, exe_args, run_args=run_args)
        run_settings.set_tasks(1)
        return run_settings

    def _build_batch_settings(self, db_nodes, batch, account, queue, time):
        batch_settings = None
        if batch:
            batch_settings = CobaltBatchSettings(
                nodes=db_nodes, time=time, queue=queue, account=account
            )
        return batch_settings

    def _initialize_entities(self, **kwargs):
        """Initialize DBNode instances for the orchestrator."""
        db_nodes = kwargs.get("db_nodes", 1)
        cluster = not bool(db_nodes < 3)
        if int(db_nodes) == 2:
            raise SSUnsupportedError(
                "CobaltOrchestrator does not support clusters of size 2"
            )
        port = kwargs.get("port", 6379)

        db_conf = CONFIG.redis_conf
        redis_exe = CONFIG.redis_exe
        ai_module = self._get_AI_module()
        start_script = self._find_redis_start_script()

        # Build DBNode instance for each node listed
        for db_id in range(db_nodes):
            db_node_name = "_".join((self.name, str(db_id)))
            start_script_args = [
                start_script,  # redis_starter.py
                f"+ifname={self._interface}",  # pass interface to start script
                "+command",  # command flag for argparser
                redis_exe,  # redis-server
                db_conf,  # redis6.conf file
                ai_module,  # redisai.so
                "--port",  # redis port
                str(port),  # port number
            ]

            if cluster:
                start_script_args += self._get_cluster_args(db_node_name, port)

            # Python because we use redis_starter.py to run redis
            run_settings = self._build_run_settings(
                "python", start_script_args, **kwargs
            )
            node = DBNode(db_node_name, self.path, run_settings, [port])
            self.entities.append(node)
        self.ports = [port]

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
        self._reserved_batch_args = [
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
