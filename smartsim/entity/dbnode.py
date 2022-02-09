# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

import os
import os.path as osp
import time

from ..error import SmartSimError
from ..log import get_logger
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
        self._mpmd = False
        self._shard_ids = None
        self._hosts = None

    @property
    def host(self):
        if not self._host:
            self._host = self._parse_db_host()
        return self._host

    @property
    def hosts(self):
        if not self._hosts:
            self._hosts = self._parse_db_hosts()
        return self._hosts

    def set_host(self, host):
        self._host = str(host)

    def set_hosts(self, hosts):
        self._hosts = [str(host) for host in hosts]

    def remove_stale_dbnode_files(self):
        """This function removes the .conf, .err, and .out files that
        have the same names used by this dbnode that may have been
        created from a previous experiment execution.
        """

        for port in self.ports:
            if not self._mpmd:
                conf_file = osp.join(self.path, self._get_cluster_conf_filename(port))
                if osp.exists(conf_file):
                    os.remove(conf_file)
            else:  # cov-lsf
                conf_files = [
                    osp.join(self.path, filename)
                    for filename in self._get_cluster_conf_filenames(port)
                ]
                for conf_file in conf_files:
                    if osp.exists(conf_file):
                        os.remove(conf_file)

        for file_ending in [".err", ".out", ".mpmd"]:
            file_name = osp.join(self.path, self.name + file_ending)
            if osp.exists(file_name):
                os.remove(file_name)
        if self._mpmd:
            for file_ending in [".err", ".out"]:
                for shard_id in self._shard_ids:
                    file_name = osp.join(
                        self.path, self.name + "_" + str(shard_id) + file_ending
                    )
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

    def _get_cluster_conf_filenames(self, port):  # cov-lsf
        """Returns the .conf file name for the given port number

        This function should bu used if and only if ``_mpmd==True``

        :param port: port number
        :type port: int
        :return: the dbnode configuration file name
        :rtype: str
        """
        return [
            "".join(("nodes-", self.name + f"_{shard_id}", "-", str(port), ".conf"))
            for shard_id in self._shard_ids
        ]

    def _parse_db_host(self):
        """Parse the database host/IP from the output file

        :raises SmartSimError: if host/ip could not be found
        :return: ip address | hostname
        :rtype: str
        """
        filepath = osp.join(self.path, self.name + ".out")
        trials = 5
        ip = None

        # try a few times to give the database files time to
        # populate on busy systems.
        while not ip and trials > 0:
            try:
                with open(filepath, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        content = line.split()
                        if "IPADDRESS:" in content:
                            ip = content[-1]
            # suppress error
            except FileNotFoundError:
                pass

            logger.debug("Waiting for Redis output files to populate...")
            if not ip:
                time.sleep(1)
                trials -= 1

        if not ip:
            logger.error("Redis IP address lookup strategy failed.")
            raise SmartSimError("Failed to obtain database hostname")

        return ip

    def _parse_db_hosts(self):
        """Parse the database hosts/IPs from the output files

        this uses the RedisIP module that is built as a dependency
        The IP address is preferred, but if hostname is only present
        then a lookup to /etc/hosts is done through the socket library.
        This function must be called only if ``_mpmd==True``.

        :raises SmartSimError: if host/ip could not be found
        :return: ip addresses | hostnames
        :rtype: list[str]
        """
        ips = []

        # Find out if all shards' output streams are piped to different file
        multiple_files = None
        for _ in range(5):
            filepath = osp.join(self.path, self.name + f"_{self._shard_ids[0]}.out")
            if osp.isfile(filepath):
                multiple_files = True
                break

            # If we did not find separate files for each shard, it could
            # be that all outputs are piped to same file OR that the separate
            # files have not been created yet. To find out whether the
            # streams are piped to the same file, we search the output file
            # for "IPADDRESS": if we find it, we can set multiple_files to False
            # and wait until the file contains enough IPs. Otherwise we
            # go to next iteration, to check if there are multiple files.
            filepath = osp.join(self.path, self.name + ".out")
            ips = []
            try:
                with open(filepath, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        content = line.split()
                        if "IPADDRESS:" in content:
                            ip = content[-1]
                            ips.append(ip)

            # suppress error
            except FileNotFoundError:
                pass

            logger.debug("Waiting for RedisIP files to populate...")
            if len(ips) < len(self._shard_ids):
                # Larger sleep time, as this seems to be needed for
                # multihost setups
                if len(ips) > 0:
                    # if we found at least one "IPADDRESS", we know
                    # output streams all go to the same file
                    multiple_files = False
                    break
                else:
                    ips = []
                    time.sleep(5)
                    continue
            else:
                ips = list(dict.fromkeys(ips))
                return ips

        if multiple_files is None:
            logger.error("RedisIP address lookup strategy failed.")
            raise SmartSimError("Failed to obtain database hostname")

        if multiple_files == True:
            for shard_id in self._shard_ids:
                trials = 5
                ip = None
                filepath = osp.join(self.path, self.name + f"_{shard_id}.out")
                # try a few times to give the database files time to
                # populate on busy systems.
                while not ip and trials > 0:
                    try:
                        with open(filepath, "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                content = line.split()
                                if "IPADDRESS:" in content:
                                    ip = content[-1]
                                    break

                    # suppress error
                    except FileNotFoundError:
                        pass

                    logger.debug("Waiting for RedisIP files to populate...")
                    if not ip:
                        # Larger sleep time, as this seems to be needed for
                        # multihost setups
                        time.sleep(5)
                        trials -= 1

                if not ip:
                    logger.error("RedisIP address lookup strategy failed.")
                    raise SmartSimError("Failed to obtain database hostname")

                ips.append(ip)

        else:
            filepath = osp.join(self.path, self.name + ".out")
            trials = 5
            ips = []
            while len(ips) < len(self._shard_ids) and trials > 0:
                ips = []
                try:
                    with open(filepath, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            content = line.split()
                            if "IPADDRESS:" in content:
                                ip = content[-1]
                                ips.append(ip)

                # suppress error
                except FileNotFoundError:
                    pass

                logger.debug("Waiting for RedisIP files to populate...")
                if len(ips) < len(self._shard_ids):
                    # Larger sleep time, as this seems to be needed for
                    # multihost setups
                    time.sleep(5)
                    trials -= 1

            if len(ips) < len(self._shard_ids):
                logger.error("RedisIP address lookup strategy failed.")
                raise SmartSimError("Failed to obtain database hostname")

        ips = list(dict.fromkeys(ips))
        return ips
