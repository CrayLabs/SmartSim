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
        gpus_per_shard=0,
        batch=True,
        hosts=None,
        project=None,
        time=None,
        db_per_host=1,
        host_map=None,
        **kwargs,
    ):

        """Initialize an Orchestrator reference for LSF based systems

        The orchestrator launches as a batch by default. If
        batch=False, at launch, the orchestrator will look for an interactive
        allocation to launch on.

        The LSFOrchestrator port provided will be incremented if multiple
        databases per host are launched (``db_per_host>1``).

        Each database shard is assigned a resource set with cpus and gpus
        allocated contiguously on the host:
        it is the user's responsibility to check if
        enough resources are available on each host.

        A list of hosts to launch the database on can be specified
        these addresses must correspond to
        those of the first ``db_nodes//db_per_host`` compute nodes
        in the allocation: for example, for 8 ``db_nodes`` and 2 ``db_per_host``
        the ``host_list`` must contain the addresses of hosts 1, 2, 3, and 4.

        ``LSFOrchestrator`` is launched with only one ``jsrun`` command
        as launch binary, and an Explicit Resource File (ERF) which is
        automatically generated. The orchestrator is always launched on the
        first ``db_nodes//db_per_host`` compute nodes in the allocation.

        Database nodes should always be accessed using high-speed networks. If
        the hostnames provided are not attached to high-speed networks,
        ``host_map`` should provide a way to obtain the high-speed
        network addresses (or hostnames). See the function ``convert_hostnames``
        for details.

        :param port: TCP/IP port, defaults to 6379
        :type port: int
        :param db_nodes: number of database shards, defaults to 1
        :type db_nodes: int, optional
        :param cpus_per_shard: cpus to allocate per shard, defaults to 4
        :type cpus_per_shard: int, optional
        :param gpus_per_shard: gpus to allocate per shard, defaults to 0
        :type gpus_per_shard: int, optional
        :param batch: Run as a batch workload, defaults to True
        :type batch: bool, optional
        :param hosts: specify hosts to launch on
        :type hosts: list[str], optional
        :param project: project to run batch on
        :type project: str, optional
        :param time: walltime for batch 'HH:MM' format
        :type time: str, optional
        :param db_per_host: number of database shards per system host (MPMD), defaults to 1
        :type db_per_host: int, optional
        :param host_map: function to convert hostnames
        :type host_map: callable function, optional
        """
        self.cpus_per_shard = cpus_per_shard
        self.gpus_per_shard = gpus_per_shard
        self.host_map = host_map

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
        self._reserved_run_args = {}
        self._reserved_batch_args = {}
        self._fill_reserved()

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

    def set_batch_arg(self, arg, value):
        """Set a cobalt ``qsub`` argument

        Some commonly used arguments are used
        by SmartSim and will not be allowed to be set.
        For example, "m", "n", etc.

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
                f"Can not set batch argument {arg}: it is a reserved keyword in LSFOrchestrator"
            )
        else:
            self.batch_settings.batch_args[arg] = value

    def set_run_arg(self, arg, value):
        """Set a run argument the orchestrator should launch
        each node with (it will be passed to `aprun`)

        Some commonly used arguments are used
        by SmartSim and will not be allowed to be set.
        For example, "chdir", "np"

        :param arg: run argument to set
        :type arg: str
        :param value: run parameter - set to None if no parameter value
        :type value: str | None
        """
        if arg in self._reserved_run_args[type(self.entities[0].run_settings)]:
            logger.warning(
                f"Can not set run argument {arg}: it is a reserved keyword in LSFOrchestrator"
            )
        else:
            for db in self.entities:
                db.run_settings.run_args[arg] = value

    def set_hosts(self, host_list):
        """Specify the hosts for the ``LSFOrchestrator`` to launch on

        :param host_list: list of host (compute node names)
        :type host_list: str | list[str]
        :raises TypeError: if wrong type
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all([isinstance(host, str) for host in host_list]):
            raise TypeError("host_list argument must be list of strings")

        if self.host_map:
            high_speed_hosts = self.convert_hostnames(host_list)
            while "" in high_speed_hosts:
                high_speed_hosts.remove("")
        else:
            high_speed_hosts = host_list

        # TODO check length
        if self.batch:
            self.batch_settings.set_hostlist(host_list)
        for db in self.entities:
            db.set_hosts(high_speed_hosts)

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
            if self.gpus_per_shard > 1:  # pragma: no-cover
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
        node = DBNode(
            self.name,
            self.path,
            run_settings,
            ports,
            self.convert_hostnames if self.host_map else None,
        )
        node._multihost = True
        node._shard_ids = range(db_nodes)
        self.entities.append(node)
        self.ports = ports

    @property
    def num_shards(self):
        return self.db_nodes

    def convert_hostnames(self, hosts):
        """Convert hostnames (or IPs) to corresponding high-speed network
        addresses.

        This is used on systems where the hosts have different
        names and addresses for the different networks they are
        attached to. For example, on some systems, the host `host1`
        can be reached over InfiniBand with the hostname `host1-IB`,
        in that case, translator should be a function returning the
        IP address of `host1-1B`. The function can also just return
        the high-speed network hostname, but the IP is preferrable.

        If and only if hostnames cannot be obtained,
        ``self.host_map`` should take IP addresses as arguments.

        If some hostnames should not be used (e.g. because they are
        batch nodes), the converter should convert them to an empty
        string.

        This function does a minimum of sanitization, avoiding
        addresses of more than 256 characters or containing spaces.

        :hosts: list of hostnames or IPs to convert
        :type hosts: list[str]
        """
        if self.host_map is None:
            return hosts
        converted_hosts = []
        for host in hosts:
            converted_host = self.host_map(host)
            if not isinstance(converted_host, str):
                raise TypeError("Converter function must return string")

            if len(converted_host) > 256:
                logger.warning(
                    "Converter returned hostname or address "
                    + f"longer than 256 characters for original hostname {host}, "
                    + "the name will be truncated"
                )
                converted_host = converted_host[:256]
            if len(converted_host.split()) > 1:
                raise ValueError(
                    f"Converter must return single value for hostname {host}, "
                    f'but returned "{converted_host}".'
                )
            converted_hosts.append(converted_host)

        return converted_hosts

    def _fill_reserved(self):
        """Fill the reserved batch and run arguments dictionaries"""
        # ERF basically makes all other args useless
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

        self._reserved_batch_args = ["J", "o", "e", "m", "n", "nnodes"]
