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

import time

from ..config import CONFIG
from ..entity import DBNode
from ..error import SmartSimError, SSUnsupportedError
from ..settings import JsrunSettings, BsubBatchSettings
from ..utils import get_logger
from .orchestrator import Orchestrator

logger = get_logger(__name__)


class LSFOrchestrator(Orchestrator):
    def __init__(
        self,
        port=6379,
        db_nodes=1,
        batch=True,
        hosts=None,
        project=None,
        time=None,
        db_per_host=1,
        force_port_increment=False,
        **kwargs,
    ):

        """Initialize an Orchestrator reference for Slurm based systems

        The orchestrator launches as a batch by default. If 
        batch=False, at launch, the orchestrator will look for an interactive
        allocation to launch on.

        The LSFOrchestrator port provided will be incremented if multiple
        databases per host are launched (``db_per_host>1``) or if
        ``force_port_increment`` is set to ``True`` (this is useful if
        the user does not know how many resource sets will be scheduled
        on each node). 
        
        Each database shard is assigned a resource set: 
        it is the user's responsibility to check if
        enough resources are available on each host. Resource sets can be
        defined by providing ``run_args`` as argument.

        LSFOrchestrator is launched with ``jsrun``
        as launch binary.

        :param port: TCP/IP port
        :type port: int
        :param db_nodes: number of database shards, defaults to 1
        :type db_nodes: int, optional
        :param batch: Run as a batch workload, defaults to True
        :type batch: bool, optional
        :param hosts: specify hosts to launch on
        :type hosts: list[str]
        :param project: project to run batch on
        :type project: str
        :param time: walltime for batch 'HH:MM' format
        :type time: str
        :param db_per_host: number of database per host, defaults to 1
        :type db_per_host: int, optional
        :param force_port_increment: whether to increment port number for
                                     each DB shard. This requires ``db_nodes``
                                     available consecutive ports on each node
        """        
        self.force_port_increment = force_port_increment
        super().__init__(
            port,
            db_nodes=db_nodes,
            batch=batch,
            run_command='jsrun',
            dpn=db_per_host,
            **kwargs,
        )
        self.db_nodes = db_nodes
        self.batch_settings = self._build_batch_settings(
            db_nodes, batch, project, time, dpn=db_per_host, **kwargs
        )
        if hosts:
            self.set_hosts(hosts)
        
    def set_cpus(self, num_cpus):
        """Set the number of CPUs available to each database shard

        This effectively will determine how many cpus can be used for
        compute threads, background threads, and network I/O.

        :param num_cpus: number of cpus to set
        :type num_cpus: int
        """
        for db in self:
            db.run_settings.set_cpus_per_rs(num_cpus)

    def set_walltime(self, walltime):
        """Set the batch walltime of the orchestrator

        Note: This will only effect orchestrators launched as a batch

        :param walltime: amount of time e.g. 10 hours is 10:00
        :type walltime: str
        :raises SmartSimError: if orchestrator isn't launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set walltime")
        self.batch_settings.set_walltime(walltime)

    def set_hosts(self, host_list):
        """Specify the hosts for the ``LSFOrchestrator`` to launch on

        :param host_list: list of host (compute node names)
        :type host_list: list[str]
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
            
    def set_batch_arg(self, arg, value):
        """Set a Bsub batch argument the orchestrator should launch with

        Some commonly used arguments are used
        by SmartSim and will not be allowed to be set.

        :param arg: batch argument to set
        :type arg: str
        :param value: batch param - set to None if no param value
        :type value: str | None
        :raises SmartSimError: if orchestrator not launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set batch_arg")
        # TODO catch commonly used arguments we use for SmartSim here
        self.batch_settings.batch_args[arg] = value

    def _build_batch_settings(self, db_nodes, batch, project, time, **kwargs):
        batch_settings = None
        dph = kwargs.get("dpn", 1)
        if batch:
            batch_settings = BsubBatchSettings(
                nodes=db_nodes//dph, time=time, project=project
            )
        return batch_settings

    def _build_run_settings(self, exe, exe_args, **kwargs):
        dph = kwargs.get("dpn", 1)
        run_args = kwargs.get("run_args", {}).copy()
        host = kwargs.get("host", "*")

        if not "nrs" in run_args.keys():
            run_args["nrs"] = 1
        if not "rs_per_host" in run_args.keys():
            run_args["rs_per_host"] = 1
        
        run_args["tasks_per_rs"] = 1
        # bind = run_args.get("cpu_per_rs", 1)
        # if bind == "ALL_CPUS":
        #     bind = 42

        run_args["launch_distribution"] = "packed"
        run_settings = JsrunSettings(exe, exe_args, run_args=run_args)
        run_settings.set_binding("none")
        # tell step to create a mpmd executable even if we only have one task
        # because we need to specify the host
        run_settings.set_mpmd_args(smts_per_task=42//dph, host=host)
        return run_settings

    def _initialize_entities(self, **kwargs):
        """Initialize DBNode instances for the orchestrator."""
        db_nodes = kwargs.get("db_nodes", 1)
        cluster = not bool(db_nodes < 3)
        if int(db_nodes) == 2:
            raise SSUnsupportedError("Orchestrator does not support clusters of size 2")

        dph = kwargs.get("dpn", 1)
        port = kwargs.get("port", 6379)

        db_conf = CONFIG.redis_conf
        exe = CONFIG.redis_exe
        ip_module = self._get_IP_module_path()
        ai_module = self._get_AI_module()

        for db_id in range(db_nodes//dph):
            db_node_name = "_".join((self.name, str(db_id)))
            # create the exe_args list for launching multiple databases
            # per node. also collect port range for dbnode
            if self.force_port_increment:
                base_port = int(port) + db_id*dph
            else:
                base_port = int(port)
            ports = []
            exe_args = []
            for port_offset in range(dph):
                next_port = base_port + port_offset
                node_exe_args = [
                    db_conf,
                    ai_module,
                    ip_module,
                    "--port",
                    str(next_port),
                ]
                if cluster:
                    node_exe_args += self._get_cluster_args(db_node_name, next_port)
                exe_args.append(node_exe_args)
                ports.append(next_port)

            # host=db_id+1 because 0 is login node
            run_settings = self._build_run_settings(exe, exe_args, host=db_id+1, **kwargs)
            node = DBNode(db_node_name, self.path, run_settings, ports)
            self.entities.append(node)
        self.ports = ports

    def __len__(self):
        return self.db_nodes