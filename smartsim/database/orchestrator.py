import os
import time

from rediscluster import RedisCluster
from rediscluster.exceptions import ClusterDownError

import socket
from os import getcwd
import os.path as osp
from ..entity import DBNode, EntityList
from ..error import SSConfigError, SmartSimError
from ..utils import get_logger
from ..utils.helpers import get_env
from ..utils.helpers import expand_exe_path
from ..settings.settings import RunSettings
from ..launcher.util.shell import execute_cmd

logger = get_logger(__name__)


class Orchestrator(EntityList):
    """The Orchestrator is an in-memory database that can be launched
    alongside entities in SmartSim. Data can be transferred between
    entities by using one of the Python, C, C++ or Fortran clients
    within an entity.
    """

    def __init__(self, start_port=6379, db_nodes=1, **kwargs):
        """Initialize an Orchestrator reference for local launch

        :param start_port: TCP/IP port
        :type start_port: int
        :param db_nodes: number of database shards, defaults to 1
        :type db_nodes: int, optional

        Extra configurations for RedisAI

        See https://oss.redislabs.com/redisai/configuration/

        :param threads_per_queue: threads per GPU device
        :type threads_per_queue: int
        :param inter_op_threads: threads accross CPU operations
        :type inter_op_threads: int
        :param intra_op_threads: threads per CPU operation
        :type intra_op_threads: int
        """
        # TODO take out DbNodes here..
        self.ports = []
        self._hosts = []
        self.path = getcwd()
        self.queue_threads = kwargs.get("threads_per_queue", None)
        self.inter_threads = kwargs.get("inter_op_threads", None)
        self.intra_threads = kwargs.get("intra_op_threads", None)
        super().__init__("orchestrator",
                         self.path,
                         db_nodes=db_nodes,
                         port=start_port,
                         **kwargs)

    @property
    def hosts(self):
        """Return the hostnames of orchestrator instance hosts

        Note that this will only be popluated after the orchestrator
        has been launched by SmartSim.

        :return: hostnames
        :rtype: list[str]
        """
        #TODO test if active?
        if not self._hosts:
            self._hosts = self._get_db_hosts()
        return self._hosts

    def remove_stale_files(self):
        """Can be used to remove database files of a previous launch"""

        for dbnode in self.entities:
            dbnode.remove_stale_dbnode_files()

    def create_cluster(self):
        """Connect launched cluster instances.

        Should only be used in the case where cluster initilization
        needs to occur manually.
        :raises SmartSimError: if cluster creation fails
        """
        #TODO check for cluster already being created.
        ip_list = []
        for host in self.hosts:
            ip = get_ip_from_host(host)
            for port in self.ports:
                address = ":".join((ip, str(port) + " "))
                ip_list.append(address)

        # TODO make a get redis_cli function
        # call cluster command
        smartsimhome = get_env("SMARTSIMHOME")
        redis_cli = os.path.join(smartsimhome, "third-party/redis/src/redis-cli")
        cmd = [redis_cli, "--cluster", "create"]
        cmd += ip_list
        cmd +=["--cluster-replicas", "0"]
        returncode, out, err = execute_cmd(cmd, proc_input="yes", shell=False)

        if returncode != 0:
            logger.error(out)
            logger.error(err)
            raise SmartSimError("Database '--cluster create' command failed")
        logger.debug(out)

        # Ensure cluster has been setup correctly
        self.check_cluster_status()
        logger.info(f"Database cluster created with {str(len(self.hosts))} shards")


    def check_cluster_status(self):
        """Check that a cluster is up and running

        :raises SmartSimError: If cluster status cannot be verified
        """
        #TODO use silc for this, then we don't have to create host dictionary
        host_list = []
        for host in self.hosts:
            for port in self.ports:
                host_dict = dict()
                host_dict["host"] = host
                host_dict["port"] = port
                host_list.append(host_dict)

        trials = 10
        logger.debug("Beginning database cluster status check...")
        while trials > 0:
            # wait for cluster to spin up
            time.sleep(2)
            try:
                redis_tester = RedisCluster(startup_nodes=host_list)
                redis_tester.set("__test__", "__test__")
                redis_tester.delete("__test__")
                logger.debug("Cluster status verified")
                return
            except ClusterDownError:
                logger.debug("Cluster still spinning up...")
                time.sleep(5)
                trials -= 1
        if trials == 0:
            raise SmartSimError("Cluster setup could not be verified")

    def _get_AI_module(self):
        """Get the RedisAI module from third-party installations

        :raises SSConfigError: if retrieval fails
        :return: path to module
        :rtype: str
        """
        sshome = get_env("SMARTSIMHOME")
        gpu_module = osp.join(sshome, "third-party/RedisAI/install-gpu/redisai.so")
        cpu_module = osp.join(sshome, "third-party/RedisAI/install-cpu/redisai.so")

        module = ["--loadmodule"]
        # if built for GPU
        if osp.isfile(gpu_module):
            logger.debug("Orchestrator using RedisAI GPU")
            module.append(gpu_module)
            if self.queue_threads:
                module.append(f"THREADS_PER_QUEUE {self.queue_threads}")
            return " ".join(module)
        if osp.isfile(cpu_module):
            logger.debug("Orchestrator using RedisAI CPU")
            module.append(cpu_module)
            if self.inter_threads:
                module.append(f"INTER_OP_THREADS {self.inter_threads}")
            if self.intra_threads:
                module.append(f"INTRA_OP_THREADS {self.intra_threads}")
            return " ".join(module)
        raise SSConfigError("Could not find RedisAI module")

    @staticmethod
    def _get_IP_module_path():
        """Get the RedisIP module from third-party installations

        :raises SSConfigError: if retrieval fails
        :return: path to module
        :rtype: str
        """
        sshome = get_env("SMARTSIMHOME")
        module_path = osp.join(sshome, "third-party/RedisIP/build/libredisip.so")
        if not osp.isfile(module_path):
            msg = "Could not locate RedisIP module.\n"
            msg += f"looked at path {module_path}"
            raise SSConfigError(msg)
        return " ".join(("--loadmodule", module_path))


    @staticmethod
    def _get_db_config_path():
        """Find the database configuration file on the filesystem

        :raises SSConfigError: if env not setup for SmartSim
        :return: path to configuration file for the database
        :rtype: str
        """
        sshome = get_env("SMARTSIMHOME")
        conf_path = osp.join(sshome, "smartsim/database/redis6.conf")
        if not osp.isfile(conf_path):
            msg = "Could not locate database configuration file.\n"
            msg += f"looked at path {conf_path}"
            raise SSConfigError(msg)
        return conf_path


    def _initialize_entities(self, **kwargs):
        """Initialize DBNode instances for the orchestrator.
        """
        dpn = kwargs.get("dpn", 1)
        db_nodes = kwargs.get("db_nodes", 1)
        port = kwargs.get("port", 6379)
        cluster = not bool(db_nodes < 3)

        db_conf = self._get_db_config_path()
        ip_module = self._get_IP_module_path()
        ai_module = self._get_AI_module()
        exe = self._find_db_exe()

        for db_id in range(db_nodes):
            db_node_name = "_".join((self.name, str(db_id)))
            # create the exe_args list for launching multiple databases
            # per node. also collect port range for dbnode
            ports = []
            exe_args = []
            for port_offset in range(dpn):
                next_port = int(port) + port_offset
                node_exe_args = [db_conf, ai_module, ip_module, "--port", str(next_port)]
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

    @staticmethod
    def _find_db_exe():
        """Find the database executable for the orchestrator

        :raises SSConfigError: if env not setup for SmartSim
        :return: path to database exe
        :rtype: str
        """
        sshome = get_env("SMARTSIMHOME")
        exe = osp.join(sshome, "third-party/redis/src/redis-server")
        try:
            full_exe = expand_exe_path(exe)
            return full_exe
        except SSConfigError:
            msg = "Database not built/installed correctly. "
            msg += "Could not locate database executable"
            raise SSConfigError(msg) from None

    @staticmethod
    def _get_cluster_args( name, port):
        """Create the arguments neccessary for cluster creation
        """
        cluster_conf =  "".join(("nodes-", name, "-", str(port), ".conf"))
        db_args = ["--cluster-enabled yes", "--cluster-config-file", cluster_conf]
        return db_args

    def _build_run_settings(self, exe, exe_args, **kwargs):
        """Build run settings for the orchestrator when launching with
        the local launcher
        """
        dpn = kwargs.get("dpn", 1)
        if dpn > 1:
            raise SSConfigError(
                "Local launching does not support multiple databases per node (MPMD)")
        run_settings = RunSettings(exe, exe_args)
        return run_settings

    def _get_db_hosts(self):
        hosts = []
        for dbnode in self.entities:
            hosts.append(dbnode.host)
        return hosts


def get_ip_from_host(host):
    """Return the IP address for the interconnect.

    :param str host: hostname of the compute node e.g. nid00004
    :returns: ip of host
    :rtype: str
    """
    ip_address = socket.gethostbyname(host)
    return ip_address
