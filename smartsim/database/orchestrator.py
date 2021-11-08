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
import logging
import os
import socket
import time
from os import getcwd
from pathlib import Path

import psutil
import redis
from rediscluster import RedisCluster
from rediscluster.exceptions import ClusterDownError, RedisClusterException
from smartredis.error import RedisReplyError

logging.getLogger("rediscluster").setLevel(logging.WARNING)

from smartredis import Client

from ..config import CONFIG
from ..entity import DBNode, EntityList
from ..error import SmartSimError
from ..launcher.util.shell import execute_cmd
from ..settings.settings import RunSettings
from ..utils import get_logger

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

    def create_cluster(self):  # cov-wlm
        """Connect launched cluster instances.

        Should only be used in the case where cluster initialization
        needs to occur manually which is not often.

        :raises SmartSimError: if cluster creation fails
        """
        ip_list = []
        for host in self.hosts:
            ip = get_ip_from_host(host)
            for port in self.ports:
                address = ":".join((ip, str(port) + " "))
                ip_list.append(address)

        # call cluster command
        redis_cli = CONFIG.redis_cli
        cmd = [redis_cli, "--cluster", "create"]
        cmd += ip_list
        cmd += ["--cluster-replicas", "0"]
        returncode, out, err = execute_cmd(cmd, proc_input="yes", shell=False)

        if returncode != 0:
            logger.error(out)
            logger.error(err)
            raise SmartSimError("Database '--cluster create' command failed")
        logger.debug(out)

        # Ensure cluster has been setup correctly
        self.check_cluster_status()
        logger.info(f"Database cluster created with {self.num_shards} shards")

    def check_cluster_status(self, trials=10):  # cov-wlm
        """Check that a cluster is up and running
        :param trials: number of attempts to verify cluster status
        :type trials: int, optional
        :raises SmartSimError: If cluster status cannot be verified
        """
        host_list = []
        for host in self.hosts:
            for port in self.ports:
                host_dict = dict()
                host_dict["host"] = get_ip_from_host(host)
                host_dict["port"] = port
                host_list.append(host_dict)

        logger.debug("Beginning database cluster status check...")
        while trials > 0:
            # wait for cluster to spin up
            time.sleep(5)
            try:
                redis_tester = RedisCluster(startup_nodes=host_list)
                redis_tester.set("__test__", "__test__")
                redis_tester.delete("__test__")
                logger.debug("Cluster status verified")
                return
            except (ClusterDownError, RedisClusterException, redis.RedisError):
                logger.debug("Cluster still spinning up...")
                trials -= 1
        if trials == 0:
            raise SmartSimError("Cluster setup could not be verified")

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
            except redis.RedisError:
                return False
        # if a cluster
        else:
            try:
                self.check_cluster_status(trials=1)
                return True
            except SmartSimError:
                return False

    def _get_AI_module(self):
        """Get the RedisAI module from third-party installations

        :raises SSConfigError: if retrieval fails
        :return: path to module
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

    def _initialize_entities(self, **kwargs):
        port = kwargs.get("port", 6379)

        db_per_host = kwargs.get("db_per_host", 1)
        if db_per_host > 1:
            raise SmartSimError(
                "Local Orchestrator does not support multiple databases per node (MPMD)"
            )
        db_nodes = kwargs.get("db_nodes", 1)
        if db_nodes > 1:
            raise SmartSimError(
                "Local Orchestrator does not support multiple database shards"
            )

        # collect database launch command information
        db_conf = CONFIG.redis_conf
        redis_exe = CONFIG.redis_exe
        ai_module = self._get_AI_module()
        start_script = self._find_redis_start_script()

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

    @staticmethod
    def _find_redis_start_script():
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        script_path = current_dir.joinpath("redis_starter.py").resolve()
        return str(script_path)

    def _check_network_interface(self):
        net_if_addrs = psutil.net_if_addrs()
        if self._interface not in net_if_addrs and self._interface != "lo":
            available = list(net_if_addrs.keys())
            logger.warning(
                f"{self._interface} is not a valid network interface on this node. \n"
                "This could be because the head node doesn't have the same networks, if so, ignore this."
            )
            logger.warning(f"Found network interfaces are: {available}")

    def enable_checkpoints(self, frequency):
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

    def set_max_memory(self, mem):
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

    def set_eviction_strategy(self, strategy):
        """Sets how the database will select what to remove when
        'maxmemory' is reached. The default is noeviction.

        :param strategy: The max memory policy to use e.g. "volatile-lru", "allkeys-lru", etc.
        :type strategy: str

        :raises SmartSimError: If 'strategy' is an invalid maxmemory policy
        :raises SmartSimError: If database is not active
        """
        self.set_db_conf("maxmemory-policy", strategy)

    def set_max_clients(self, clients=50_000):
        """Sets the max number of connected clients at the same time.
        When the number of DB shards contained in the orchestrator is
        more than two, then every node will use two connections, one
        incoming and another outgoing.

        :param clients: the maximum number of connected clients
        :type clients: int, optional
        """
        self.set_db_conf("maxclients", str(clients))

    def set_max_message_size(self, size=1_073_741_824):
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

    def set_db_conf(self, key, value):
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
                raise SmartSimError(
                    "Incompatible function arguments. The key and value used for setting the database configurations must be strings."
                ) from None
        else:
            raise SmartSimError(
                "The SmartSim Orchestrator must be active in order to set the database's configurations."
            )


def get_ip_from_host(host):
    """Return the IP address for the interconnect.

    :param host: hostname of the compute node e.g. nid00004
    :type host: str
    :returns: ip of host
    :rtype: str
    """
    ip_address = socket.gethostbyname(host)
    return ip_address
