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
from ..settings import MpirunSettings, SbatchSettings, SrunSettings
from ..utils import get_logger
from .orchestrator import Orchestrator

logger = get_logger(__name__)


class SlurmOrchestrator(Orchestrator):
    def __init__(
        self,
        port=6379,
        db_nodes=1,
        batch=True,
        hosts=None,
        run_command="srun",
        account=None,
        time=None,
        alloc=None,
        dpn=1,
        **kwargs,
    ):

        """Initialize an Orchestrator reference for Slurm based systems

        The orchestrator launches as a batch by default. The Slurm orchestrator
        can also be given an allocation to run on. If no allocation is provided,
        and batch=False, at launch, the orchestrator will look for an interactive
        allocation to launch on.

        The SlurmOrchestrator port provided will be incremented if multiple
        databases per node are launched.

        SlurmOrchestrator supports launching with both ``srun`` and ``mpirun``
        as launch binaries. If mpirun is used, the hosts parameter should be
        populated with length equal to that of the ``db_nodes`` argument.

        :param port: TCP/IP port
        :type port: int
        :param db_nodes: number of database shards, defaults to 1
        :type db_nodes: int, optional
        :param batch: Run as a batch workload, defaults to True
        :type batch: bool, optional
        :param hosts: specify hosts to launch on
        :type hosts: list[str]
        :param run_command: specify launch binary. Options are ``mpirun`` and ``srun``
        :type run_command: str
        :param account: account to run batch on
        :type account: str
        :param time: walltime for batch 'HH:MM:SS' format
        :type time: str
        :param alloc: allocation to launch on, defaults to None
        :type alloc: str, optional
        :param dpn: number of database per node (MPMD), defaults to 1
        :type dpn: int, optional
        """
        super().__init__(
            port,
            db_nodes=db_nodes,
            batch=batch,
            run_command=run_command,
            alloc=alloc,
            dpn=dpn,
            **kwargs,
        )
        self.batch_settings = self._build_batch_settings(
            db_nodes, alloc, batch, account, time, **kwargs
        )
        if hosts:
            self.set_hosts(hosts)
        elif not hosts and run_command == "mpirun":
            raise SmartSimError(
                "hosts argument is required when launching SlurmOrchestrator with mpirun"
            )

    def set_cpus(self, num_cpus):
        """Set the number of CPUs available to each database shard

        This effectively will determine how many cpus can be used for
        compute threads, background threads, and network I/O.

        :param num_cpus: number of cpus to set
        :type num_cpus: int
        """
        if self.batch:
            self.batch_settings.batch_args["cpus-per-task"] = num_cpus
        for db in self:
            db.run_settings.set_cpus_per_task(num_cpus)

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
        """Specify the hosts for the ``SlurmOrchestrator`` to launch on

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
            db.run_settings.set_hostlist([host])

    def set_batch_arg(self, arg, value):
        """Set a Sbatch argument the orchestrator should launch with

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
        # TODO catch commonly used arguments we use for SmartSim here
        self.batch_settings.batch_args[arg] = value

    def _build_batch_settings(self, db_nodes, alloc, batch, account, time, **kwargs):
        batch_settings = None
        dpn = kwargs.get("dpn", 1)
        # enter this conditional if user has not specified an allocation to run
        # on or if user specified batch=False (alloc will be found through env)
        if not alloc and batch:
            batch_args = {"ntasks-per-node": dpn}
            batch_settings = SbatchSettings(
                nodes=db_nodes, time=time, account=account, batch_args=batch_args
            )
        return batch_settings

    def _build_run_settings(self, exe, exe_args, **kwargs):
        run_command = kwargs.get("run_command", "srun")
        if run_command == "srun":
            return self._build_srun_settings(exe, exe_args, **kwargs)
        if run_command == "mpirun":
            return self._build_mpirun_settings(exe, exe_args, **kwargs)
        raise SSUnsupportedError(
            f"SlurmOrchestrator does not support {run_command} as a launch binary"
        )

    def _build_srun_settings(self, exe, exe_args, **kwargs):
        alloc = kwargs.get("alloc", None)
        dpn = kwargs.get("dpn", 1)
        run_args = kwargs.get("run_args", {})

        # if user specified batch=False
        # also handles batch=False and alloc=False (alloc will be found by launcher)
        run_args["nodes"] = 1
        run_args["ntasks"] = dpn
        run_args["ntasks-per-node"] = dpn
        run_settings = SrunSettings(exe, exe_args, run_args=run_args, alloc=alloc)
        if dpn > 1:
            # tell step to create a mpmd executable
            run_settings.mpmd = True
        return run_settings

    def _build_mpirun_settings(self, exe, exe_args, **kwargs):
        alloc = kwargs.get("alloc", None)
        dpn = kwargs.get("dpn", 1)
        if alloc:
            msg = (
                "SlurmOrchestrator using OpenMPI cannot specify allocation to launch in"
            )
            msg += "\n User must launch in interactive allocation or as batch."
            logger.warning(msg)
        if dpn > 1:
            msg = "SlurmOrchestrator does not support multiple databases per node when launching with mpirun"
            raise SmartSimError(msg)

        run_args = kwargs.get("run_args", {})
        run_settings = MpirunSettings(exe, exe_args, run_args=run_args)
        run_settings.set_tasks(1)
        return run_settings

    def _initialize_entities(self, **kwargs):
        """Initialize DBNode instances for the orchestrator."""
        db_nodes = kwargs.get("db_nodes", 1)
        cluster = not bool(db_nodes < 3)
        if int(db_nodes) == 2:
            raise SSUnsupportedError("Orchestrator does not support clusters of size 2")

        dpn = kwargs.get("dpn", 1)
        port = kwargs.get("port", 6379)

        db_conf = CONFIG.redis_conf
        exe = CONFIG.redis_exe
        ip_module = self._get_IP_module_path()
        ai_module = self._get_AI_module()

        for db_id in range(db_nodes):
            db_node_name = "_".join((self.name, str(db_id)))
            # create the exe_args list for launching multiple databases
            # per node. also collect port range for dbnode
            ports = []
            exe_args = []
            for port_offset in range(dpn):
                next_port = int(port) + port_offset
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

            # if only launching 1 dpn, we don't need a list of exe args lists
            if dpn == 1:
                exe_args = exe_args[0]
            run_settings = self._build_run_settings(exe, exe_args, **kwargs)
            node = DBNode(db_node_name, self.path, run_settings, ports)
            self.entities.append(node)
        self.ports = ports
