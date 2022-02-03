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

from shlex import split as sh_split
from warnings import warn, simplefilter

from ..entity import DBNode
from ..error import SmartSimError, SSUnsupportedError
from ..log import get_logger
from ..settings import AprunSettings, MpirunSettings, QsubBatchSettings
from .orchestrator import Orchestrator

logger = get_logger(__name__)


class PBSOrchestrator(Orchestrator):
    def __init__(
        self,
        port=6379,
        db_nodes=1,
        batch=True,
        hosts=None,
        run_command="aprun",
        interface="ipogif0",
        account=None,
        time=None,
        queue=None,
        single_cmd=True,
        **kwargs,
    ):
        """Initialize an Orchestrator reference for PBSPro based systems

        The ``PBSOrchestrator`` launches as a batch by default. If batch=False,
        at launch, the ``PBSOrchestrator`` will look for an interactive
        allocation to launch on.

        The PBS orchestrator does not support multiple databases per node.

        If ``mpirun`` is specifed as the ``run_command``, then the ``hosts``
        argument is required.

        :param port: TCP/IP port
        :type port: int
        :param db_nodes: number of compute nodes to span accross, defaults to 1
        :type db_nodes: int, optional
        :param batch: run as a batch workload, defaults to True
        :type batch: bool, optional
        :param hosts: specify hosts to launch on, defaults to None
        :type hosts: list[str]
        :param run_command: specify launch binary. Options are ``mpirun`` and ``aprun``, defaults to "aprun"
        :type run_command: str, optional
        :param interface: network interface to use, defaults to "ipogif0"
        :type interface: str, optional
        :param account: account to run batch on
        :type account: str, optional
        :param time: walltime for batch 'HH:MM:SS' format
        :type time: str, optional
        :param queue: queue to launch batch in
        :type queue: str, optional
        """
        simplefilter('always', DeprecationWarning)
        msg = "PBSOrchestrator(...) is deprecated and will be removed in a future release.\n" 
        msg += "Please update your code to use Orchestrator(launcher='pbs', ...)."
        warn(msg, DeprecationWarning)
        super().__init__(
            port,
            interface,
            db_nodes=db_nodes,
            batch=batch,
            run_command=run_command,
            single_cmd=single_cmd,
            launcher="pbs",
            hosts=hosts,
            account=account,
            queue=queue,
            time=time,
            **kwargs,
        )
        # self.batch_settings = self._build_batch_settings(
        #     db_nodes, batch, account, time, queue
        # )
        # if hosts:
        #     self.set_hosts(hosts)
        # elif not hosts and run_command == "mpirun":
        #     raise SmartSimError(
        #         "hosts argument is required when launching PBSOrchestrator with OpenMPI"
        #     )
        # self._reserved_run_args = {}
        # self._reserved_batch_args = {}
        # self._fill_reserved()

    # def set_cpus(self, num_cpus):
    #     """Set the number of CPUs available to each database shard

    #     This effectively will determine how many cpus can be used for
    #     compute threads, background threads, and network I/O.

    #     :param num_cpus: number of cpus to set
    #     :type num_cpus: int
    #     """
    #     # if batched, make sure we have enough cpus
    #     if self.batch:
    #         self.batch_settings.set_ncpus(num_cpus)
    #     for db in self:
    #         db.run_settings.set_cpus_per_task(num_cpus)

    # def set_walltime(self, walltime):
    #     """Set the batch walltime of the orchestrator

    #     Note: This will only effect orchestrators launched as a batch

    #     :param walltime: amount of time e.g. 10 hours is 10:00:00
    #     :type walltime: str
    #     :raises SmartSimError: if orchestrator isn't launching as batch
    #     """
    #     if not self.batch:
    #         raise SmartSimError("Not running in batch, cannot set walltime")
    #     self.batch_settings.set_walltime(walltime)

    # def set_hosts(self, host_list):
    #     """Specify the hosts for the ``PBSOrchestrator`` to launch on

    #     :param host_list: list of hosts (compute node names)
    #     :type host_list: str | list[str]
    #     :raises TypeError: if host_list is wrong type
    #     """
    #     if isinstance(host_list, str):
    #         host_list = [host_list.strip()]
    #     if not isinstance(host_list, list):
    #         raise TypeError("host_list argument must be a list of strings")
    #     if not all([isinstance(host, str) for host in host_list]):
    #         raise TypeError("host_list argument must be list of strings")
    #     # TODO check length
    #     if self.batch:
    #         self.batch_settings.set_hostlist(host_list)
    #     for host, db in zip(host_list, self.entities):
    #         #db.set_host(host)

    #         # Aprun doesn't like settings hosts in batch launch
    #         if isinstance(db.run_settings, AprunSettings):
    #             if not self.batch:
    #                 db.run_settings.set_hostlist([host])
    #         else:
    #             db.run_settings.set_hostlist([host])

    #         if db._mpmd:
    #             for i, mpmd_runsettings in enumerate(db.run_settings.mpmd):
    #                 mpmd_runsettings.set_hostlist(host_list[i+1])

    # def set_batch_arg(self, arg, value):
    #     """Set a ``qsub`` argument the ``PBSOrchestrator`` should launch with

    #     Some commonly used arguments such as -e are used
    #     by SmartSim and will not be allowed to be set.

    #     :param arg: batch argument to set e.g. "A" for account
    #     :type arg: str
    #     :param value: batch param - set to None if no param value
    #     :type value: str | None
    #     :raises SmartSimError: if orchestrator not launching as batch
    #     """
    #     if not self.batch:
    #         raise SmartSimError("Not running as batch, cannot set batch_arg")

    #     if arg in self._reserved_batch_args:
    #         logger.warning(
    #             f"Can not set batch argument {arg}: it is a reserved keyword in the PBSOrchestrator"
    #         )
    #     else:
    #         self.batch_settings.batch_args[arg] = value

    # def set_run_arg(self, arg, value):
    #     """Set a run argument the orchestrator should launch
    #     each node with (it will be passed to `aprun`)

    #     Some commonly used arguments are used
    #     by SmartSim and will not be allowed to be set.

    #     :param arg: run argument to set
    #     :type arg: str
    #     :param value: run parameter - set to None if no parameter value
    #     :type value: str | None
    #     """
    #     if arg in self._reserved_run_args[type(self.entities[0].run_settings)]:
    #         logger.warning(
    #             f"Can not set run argument {arg}: it is a reserved keyword in PBSOrchestrator"
    #         )
    #     else:
    #         for db in self.entities:
    #             db.run_settings.run_args[arg] = value

    # def _build_run_settings(self, exe, exe_args, **kwargs):
    #     run_command = kwargs.get("run_command", "aprun")
    #     if run_command == "aprun":
    #         return self._build_aprun_settings(exe, exe_args, **kwargs)
    #     if run_command == "mpirun":
    #         return self._build_mpirun_settings(exe, exe_args, **kwargs)
    #     raise SSUnsupportedError(
    #         f"PBSOrchestrator does not support {run_command} as a launch binary"
    #     )

    # def _build_aprun_settings(self, exe, exe_args, **kwargs):
    #     run_args = kwargs.get("run_args", {})
    #     db_nodes = kwargs.get("db_nodes", 1)
    #     single_cmd = kwargs.get("single_cmd", True)
    #     mpmd_nodes = single_cmd and db_nodes>1

    #     if not mpmd_nodes:
    #         run_settings = AprunSettings(exe, exe_args, run_args=run_args)
    #         run_settings.set_tasks(1)
    #         run_settings.set_tasks_per_node(1)
    #     else:
    #         run_settings = AprunSettings(exe, exe_args[0], run_args=run_args)
    #         run_settings.set_tasks(1)
    #         run_settings.set_tasks_per_node(1)
    #         for exe_arg in exe_args[1:]:
    #             mpmd_run_settings = AprunSettings(exe, exe_arg, run_args.copy())
    #             mpmd_run_settings.set_tasks(1)
    #             mpmd_run_settings.set_tasks_per_node(1)
    #             run_settings.make_mpmd(mpmd_run_settings)
    #     return run_settings

    # def _build_mpirun_settings(self, exe, exe_args, **kwargs):
    #     run_args = kwargs.get("run_args", {})
    #     db_nodes = kwargs.get("db_nodes", 1)
    #     single_cmd = kwargs.get("single_cmd", True)
    #     mpmd_nodes = single_cmd and db_nodes>1

    #     if not mpmd_nodes:
    #         run_settings = MpirunSettings(exe, exe_args, run_args=run_args)
    #         run_settings.set_tasks(1)
    #     else:
    #         # tell step to create a mpmd executable
    #         run_settings = MpirunSettings(exe, exe_args[0], run_args=run_args.copy())
    #         run_settings.set_tasks(1)
    #         for exe_arg in exe_args[1:]:
    #             mpmd_run_settings = MpirunSettings(exe, exe_arg, run_args.copy())
    #             mpmd_run_settings.set_tasks(1)
    #             run_settings.make_mpmd(mpmd_run_settings)
    #     return run_settings

    # def _build_batch_settings(self, db_nodes, batch, account, time, queue):
    #     batch_settings = None
    #     if batch:
    #         batch_settings = QsubBatchSettings(
    #             nodes=db_nodes, ncpus=1, time=time, queue=queue, account=account
    #         )
    #     return batch_settings

    # def _initialize_entities(self, **kwargs):
    #     """Initialize DBNode instances for the orchestrator."""
    #     db_nodes = kwargs.get("db_nodes", 1)
    #     self.db_nodes = db_nodes
    #     single_cmd = kwargs.get("single_cmd", True)
    #     mpmd_nodes = single_cmd and db_nodes>1
    #     cluster = not bool(db_nodes < 3)
    #     if int(db_nodes) == 2:
    #         raise SSUnsupportedError(
    #             "PBSOrchestrator does not support clusters of size 2"
    #         )
    #     port = kwargs.get("port", 6379)

    #     if mpmd_nodes:
    #         exe_args = []

    #     # Build DBNode instance for each node listed
    #     for db_id in range(db_nodes):
    #         if not mpmd_nodes:
    #             db_node_name = "_".join((self.name, str(db_id)))
    #             db_shard_name = db_node_name
    #         else:
    #             db_node_name = self.name
    #             db_shard_name = "_".join((self.name, str(db_id)))

    #         start_script_args = [
    #             self._redis_launch_script,  # redis_starter.py
    #             f"+ifname={self._interface}",  # pass interface to start script
    #             "+command",  # command flag for argparser
    #             self._redis_exe,  # redis-server
    #             self._redis_conf,  # redis6.conf file
    #             self._rai_module,  # redisai.so
    #             "--port",  # redis port
    #             str(port),  # port number
    #         ]

    #         if cluster:
    #             start_script_args += self._get_cluster_args(db_shard_name, port)

    #         if mpmd_nodes:
    #             exe_args.append(" ".join(start_script_args))
    #         else:
    #             # Python because we use redis_starter.py to run redis
    #             run_settings = self._build_run_settings(
    #                 "python", start_script_args, **kwargs
    #             )
    #             node = DBNode(db_node_name, self.path, run_settings, [port])
    #             self.entities.append(node)

    #     if mpmd_nodes:
    #         exe_args = [sh_split(exe_arg) for exe_arg in exe_args]
    #         run_settings = self._build_run_settings("python", exe_args, **kwargs)
    #         node = DBNode(db_node_name, self.path, run_settings, [port])
    #         node._shard_ids = range(db_nodes)
    #         node._mpmd = True
    #         self.entities.append(node)

    #     self.ports = [port]

    # def _fill_reserved(self):
    #     """Fill the reserved batch and run arguments dictionaries"""
    #     self._reserved_run_args[MpirunSettings] = [
    #         "np",
    #         "N",
    #         "c",
    #         "output-filename",
    #         "n",
    #         "wdir",
    #         "wd",
    #         "host",
    #     ]
    #     self._reserved_run_args[AprunSettings] = [
    #         "pes",
    #         "n",
    #         "pes-per-node",
    #         "N",
    #         "l",
    #         "pes-per-numa-node",
    #         "S",
    #         "wdir",
    #     ]
    #     self._reserved_batch_args = ["e", "o", "N", "l"]

    # @property
    # def num_shards(self):
    #     return self.db_nodes
