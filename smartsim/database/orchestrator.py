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

import psutil
import redis

from .._core.config import CONFIG
from .._core.utils import check_cluster_status
from ..entity import DBNode, EntityList
from ..error import SmartSimError, SSConfigError, SSInternalError
from ..log import get_logger
from ..settings.base import RunSettings

logger = get_logger(__name__)


class Orchestrator(EntityList):
    """The Orchestrator is an in-memory database that can be launched
    alongside entities in SmartSim. Data can be transferred between
    entities by using one of the Python, C, C++ or Fortran clients
    within an entity.
    """

    def __init__(self, port=6379, interface="lo", **kwargs):
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
        self.ports = []
        self.path = getcwd()
        self._hosts = []
        self._interface = interface
        self._check_network_interface()
        self.queue_threads = kwargs.get("threads_per_queue", None)
        self.inter_threads = kwargs.get("inter_op_threads", None)
        self.intra_threads = kwargs.get("intra_op_threads", None)
        super().__init__("orchestrator", self.path, port=port, **kwargs)

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

    @property
    def num_shards(self):
        """Return the number of DB shards contained in the orchestrator.
        This might differ from the number of ``DBNode`` objects, as each
        ``DBNode`` may start more than one shard (e.g. with MPMD).

        :returns: num_shards
        :rtype: int
        """
        return len(self)

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
                check_cluster_status(trials=1)
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

    def _initialize_entities(self, **kwargs):
        port = kwargs.get("port", 6379)

        db_per_host = kwargs.get("db_per_host", 1)
        if db_per_host > 1:
            raise ValueError(
                "Local Orchestrator does not support multiple databases per node (MPMD)"
            )
        db_nodes = kwargs.get("db_nodes", 1)
        if db_nodes > 1:
            raise ValueError(
                "Local Orchestrator does not support multiple database shards"
            )

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

        exe_args = " ".join(start_script_args)

        # python is exe because we are using redis_starter.py to start redis
        run_settings = RunSettings("python", exe_args)
        db_node_name = self.name + "_0"
        node = DBNode(db_node_name, self.path, run_settings, [port])

        # add DBNode to Orchestrator
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
            if not dbnode._multihost:
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
