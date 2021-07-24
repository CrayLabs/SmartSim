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
from ..settings import BsubBatchSettings, JsrunSettings
from ..utils import get_logger
from .orchestrator import Orchestrator

logger = get_logger(__name__)


class LSFOrchestrator(Orchestrator):
    def __init__(
        self,
        port=6379,
        db_nodes=1,
        cpus_per_shard=4,
        gpus_per_shard=1,
        batch=True,
        hosts=None,
        project=None,
        time=None,
        db_per_host=1,
        **kwargs,
    ):

        """Initialize an Orchestrator reference for LSF based systems

        The orchestrator launches as a batch by default. If
        batch=False, at launch, the orchestrator will look for an interactive
        allocation to launch on.

        The LSFOrchestrator port provided will be incremented if multiple
        databases per host are launched (``db_per_host>1``).

        Each database shard is assigned a resource set:
        it is the user's responsibility to check if
        enough resources are available on each host. Resource sets can be
        defined by providing ``run_args`` as argument.

        LSFOrchestrator is launched with only one ``jsrun`` command
        as launch binary, and a Explicit Resource File (ERF) which is
        automatically generated.

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
        :param db_per_host: number of database shards per system host (MPMD), defaults to 1
        :type db_per_host: int, optional
        """
        self.cpus_per_shard = cpus_per_shard
        self.gpus_per_shard = gpus_per_shard

        super().__init__(
            port,
            db_nodes=db_nodes,
            batch=batch,
            run_command="jsrun",
            db_per_host=db_per_host,
            **kwargs,
        )
        self.db_nodes = db_nodes
        self.batch_settings = self._build_batch_settings(
            db_nodes, batch, project, time, db_per_host=db_per_host, **kwargs
        )
        if hosts:
            self.set_hosts(hosts)

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
        for db in self.entities:
            db.set_hosts(host_list)

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
        dph = kwargs.get("db_per_host", 1)
        smts = kwargs.get("smts", 1)
        if batch:
            batch_settings = BsubBatchSettings(
                nodes=db_nodes // dph, time=time, project=project, smts=smts
            )
        return batch_settings

    def _build_run_settings(self, exe, exe_args, **kwargs):
        dph = kwargs.get("db_per_host", 1)
        run_args = kwargs.get("run_args", {}).copy()
        old_host = None
        erf_rs = None
        for shard_id, args in enumerate(exe_args):
            host = shard_id // dph
            run_args["launch_distribution"] = "packed"
            run_settings = JsrunSettings(exe, args, run_args=run_args)
            run_settings.set_binding("none")
            # tell step to create a mpmd executable even if we only have one task
            # because we need to specify the host
            if host != old_host:
                assigned_smts = 0
                assigned_gpus = 0
            old_host = host

            erf_sets = {
                "rank_count": "1",
                "host": str(1 + host),
                "cpu": "{" + f"{assigned_smts}:{self.cpus_per_shard}" + "}",
            }

            assigned_smts += self.cpus_per_shard
            if self.gpus_per_shard > 1:
                erf_sets["gpu"] = (
                    "{" + f"{assigned_gpus}-{assigned_gpus+self.gpus_per_shard-1}" + "}"
                )
            elif self.gpus_per_shard > 0:
                erf_sets["gpu"] = "{" + f"{assigned_gpus}" + "}"
            assigned_gpus += self.gpus_per_shard

            run_settings.set_erf_sets(erf_sets)

            if erf_rs:
                erf_rs.make_mpmd(run_settings)
            else:
                run_settings.make_mpmd()
                erf_rs = run_settings

        return erf_rs

    def _initialize_entities(self, **kwargs):
        """Initialize DBNode instances for the orchestrator."""
        db_nodes = kwargs.get("db_nodes", 1)
        cluster = not bool(db_nodes < 3)
        if int(db_nodes) == 2:
            raise SSUnsupportedError("Orchestrator does not support clusters of size 2")

        dph = kwargs.get("db_per_host", 1)
        port = kwargs.get("port", 6379)

        db_conf = CONFIG.redis_conf
        exe = CONFIG.redis_exe
        ip_module = self._get_IP_module_path()
        ai_module = self._get_AI_module()

        exe_args = []
        for db_id in range(db_nodes // dph):
            # create the exe_args list for launching multiple databases
            # per node. also collect port range for dbnode

            ports = []
            for port_offset in range(dph):
                next_port = int(port) + port_offset
                db_shard_name = "_".join((self.name, str(db_id * dph + port_offset)))
                node_exe_args = [
                    db_conf,
                    ai_module,
                    ip_module,
                    "--port",
                    str(next_port),
                    "--logfile",
                    db_shard_name + ".out",
                ]
                if cluster:
                    node_exe_args += self._get_cluster_args(db_shard_name, next_port)
                exe_args.append(node_exe_args)
                ports.append(next_port)

        run_settings = self._build_run_settings(exe, exe_args, **kwargs)
        node = DBNode(self.name, self.path, run_settings, ports)
        node._multihost = True
        node._shard_ids = range(db_nodes)
        self.entities.append(node)
        self.ports = ports

    @property
    def num_shards(self):
        return self.db_nodes
