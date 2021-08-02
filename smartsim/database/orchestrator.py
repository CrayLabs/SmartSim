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
import multiprocessing as mp
import socket
import time
from os import getcwd

import numpy as np
from smartredis import Client
from smartredis.error import RedisConnectionError, RedisReplyError

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

    def __init__(self, port=6379, **kwargs):
        """Initialize an Orchestrator reference for local launch

        :param port: TCP/IP port
        :type port: int

        Extra configurations for RedisAI

        See https://oss.redislabs.com/redisai/configuration/

        :param threads_per_queue: threads per GPU device
        :type threads_per_queue: int
        :param inter_op_threads: threads accross CPU operations
        :type inter_op_threads: int
        :param intra_op_threads: threads per CPU operation
        :type intra_op_threads: int
        """
        self.ports = []
        self._hosts = []
        self.path = getcwd()
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

        Note that this will only be popluated after the orchestrator
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

        Should only be used in the case where cluster initilization
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

    def check_cluster_status(self):  # cov-wlm
        """Check that a cluster is up and running

        :raises SmartSimError: If cluster status cannot be verified
        """
        trials = 10
        logger.debug("Beginning database cluster status check...")
        while trials > 0:
            # wait for cluster to spin up
            time.sleep(2)
            try:
                self.is_active()
                break
            except (RedisReplyError, RedisConnectionError):
                logger.debug("Cluster still spinning up...")
                time.sleep(3)
                trials -= 1
        if trials == 0:
            raise SmartSimError("Cluster setup could not be verified")

    def get_address(self):
        """Return database addresses

        :return: addresses
        :rtype: list[str]

        :raises SmartSimError: If database address cannot be found
        """
        if not self._hosts:
            raise SmartSimError("Could not find database address")
        elif not self.is_active():
            raise SmartSimError("Database is not active")
        return self._get_address()

    def _get_address(self):
        addresses = []
        for host, port in itertools.product(self._hosts, self.ports):
            addresses.append(":".join((host, str(port))))
        return addresses


    def is_active(self):
        """Check if database is running

        :returns: True if database is active, False otherwise
        :rtype: bool
        """
        active = False

        if not self._hosts:
            return active
        addresses = self._get_address()
        cluster = True if self.num_shards > 1 else False

        try:
            client = Client(address=addresses[0], cluster=cluster)

            # if we have more than one shard to get info on
            if cluster:
                db_info = client.get_db_cluster_info(addresses)
                for info in db_info:
                    if info["cluster_state"] != "ok":
                        return False
                active = True
            else:
                tensor = np.array([1,2])
                client.put_tensor("cluster_test", tensor)
                _ = client.get_tensor("cluster_test")
                active = True
        except (RedisConnectionError, RedisReplyError):
            return False

        return active


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
            module.append(f"INTER_OP_THREADS {self.inter_threads}")
        if self.intra_threads:
            module.append(f"INTRA_OP_THREADS {self.intra_threads}")
        return " ".join(module)

    @staticmethod
    def _get_IP_module_path():
        """Get the RedisIP module from third-party installations

        :raises SSConfigError: if retrieval fails
        :return: path to module
        :rtype: str
        """
        module_path = CONFIG.redisip
        return " ".join(("--loadmodule", module_path))

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
        exe = CONFIG.redis_exe
        ip_module = self._get_IP_module_path()
        ai_module = self._get_AI_module()

        # create single DBNode instance for Local Orchestrator
        exe_args = [db_conf, ai_module, ip_module, "--port", str(port)]
        run_settings = RunSettings(exe, exe_args)
        db_node_name = self.name + "_0"
        node = DBNode(db_node_name, self.path, run_settings, [port])

        # add DBNode to Orchestrator
        self.entities.append(node)
        self.ports = [port]

    @staticmethod
    def _get_cluster_args(name, port):
        """Create the arguments neccessary for cluster creation"""
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


# Hack to avoid a bug in SmartRedis 1.1 where
# client will segfault because of a uncaught error
class ClientThread(mp.Process):
    def __init__(self, address, cluster=False):
        mp.Process.__init__(self, name="ClientThread")
        self.address = address
        self.cluster = cluster

    def run(self):
        client = Client(self.address, cluster=self.cluster)
        client.put_tensor("db_test", np.array([1]))
        receive_tensor = client.get_tensor("db_test")


def get_ip_from_host(host):
    """Return the IP address for the interconnect.

    :param str host: hostname of the compute node e.g. nid00004
    :returns: ip of host
    :rtype: str
    """
    ip_address = socket.gethostbyname(host)
    return ip_address
