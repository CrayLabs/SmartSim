import os
import os.path as osp
import time

from smartsim.error.errors import SmartSimError

from ..utils import get_logger
from .entity import SmartSimEntity

logger = get_logger(__name__)


class DBNode(SmartSimEntity):
    """DBNode objects are the entities that make up the orchestrator.
    Each database node can be launched in a cluster configuration
    and take launch multiple databases per node.

    To configure how each instance of the database operates, look
    into the smartsimdb.conf.
    """

    def __init__(self, name, path, run_settings, ports):
        """Initialize a database node within an orchestrator."""
        self.ports = ports
        self._host = None
        super().__init__(name, path, run_settings)

    @property
    def host(self):
        if not self._host:
            self._host = self._parse_db_host()
        return self._host

    def set_host(self, host):
        self._host = str(host)

    def remove_stale_dbnode_files(self):
        """This function removes the .conf, .err, and .out files that
        have the same names used by this dbnode that may have been
        created from a previous experiment execution.
        """

        for port in self.ports:
            conf_file = osp.join(self.path, self._get_cluster_conf_filename(port))
            if osp.exists(conf_file):
                os.remove(conf_file)

        for file_ending in [".err", ".out", ".mpmd"]:
            file_name = osp.join(self.path, self.name + file_ending)
            if osp.exists(file_name):
                os.remove(file_name)

    def _get_cluster_conf_filename(self, port):
        """Returns the .conf file name for the given port number

        :param port: port number
        :type port: int
        :return: the dbnode configuration file name
        :rtype: str
        """
        return "".join(("nodes-", self.name, "-", str(port), ".conf"))

    def _parse_db_host(self):
        """Parse the database host/IP from the output file

        this uses the RedisIP module that is built as a dependency
        The IP address is preferred, but if hostname is only present
        then a lookup to /etc/hosts is done through the socket library

        :raises SmartSimError: if host/ip could not be found
        :return: ip address | hostname
        :rtype: str
        """
        filepath = osp.join(self.path, self.name + ".out")
        trials = 5
        host = None
        ip = None

        # try a few times to give the database files time to
        # populate on busy systems.
        while not host and trials > 0:
            try:
                with open(filepath, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        content = line.split()
                        if "Hostname:" in content:
                            host = content[-1]
                        if "IP:" in content:
                            ip = content[-1]
                            break
            except FileNotFoundError:
                logger.debug("Waiting for RedisIP files to populate...")
                trials -= 1
                time.sleep(5)
            logger.debug("Waiting for RedisIP files to populate...")
            trials -= 1
            time.sleep(5)

        if not host and not ip:
            logger.error("RedisIP address lookup strategy failed.")
            raise SmartSimError("Failed to obtain database hostname")

        # prefer the ip address if present
        # TODO do socket lookup and ensure IP address matches
        # in the case where compute node returns 127.0.0.1 for its
        # own IP address
        if host and ip:
            host = ip
        return host
