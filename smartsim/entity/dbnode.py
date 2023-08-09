# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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
import typing as t

from ..error import SmartSimError
from ..log import get_logger
from .entity import SmartSimEntity
from ..settings.base import RunSettings


logger = get_logger(__name__)


class DBNode(SmartSimEntity):
    """DBNode objects are the entities that make up the orchestrator.
    Each database node can be launched in a cluster configuration
    and take launch multiple databases per node.

    To configure how each instance of the database operates, look
    into the smartsimdb.conf.
    """

    def __init__(
        self,
        name: str,
        path: str,
        run_settings: RunSettings,
        ports: t.List[int],
        output_files: t.List[str],
    ) -> None:
        """Initialize a database node within an orchestrator."""
        self.ports = ports
        self._host: t.Optional[str] = None
        super().__init__(name, path, run_settings)
        self._mpmd = False
        self._num_shards: int = 0
        self._hosts: t.Optional[t.List[str]] = None

        if not output_files:
            raise ValueError("output_files cannot be empty")
        if not isinstance(output_files, list) or not all(
            isinstance(item, str) for item in output_files
        ):
            raise ValueError("output_files must be of type list[str]")
        self._output_files = output_files

    @property
    def num_shards(self) -> int:
        return self._num_shards

    @num_shards.setter
    def num_shards(self, value: int) -> None:
        self._num_shards = value

    @property
    def host(self) -> str:
        if not self._host:
            self._host = self._parse_db_host()
        return self._host

    @property
    def hosts(self) -> t.List[str]:
        if not self._hosts:
            self._hosts = self._parse_db_hosts()
        return self._hosts

    @property
    def is_mpmd(self) -> bool:
        return self._mpmd

    @is_mpmd.setter
    def is_mpmd(self, value: bool) -> None:
        self._mpmd = value

    def set_host(self, host: str) -> None:
        self._host = str(host)

    def set_hosts(self, hosts: t.List[str]) -> None:
        self._hosts = [str(host) for host in hosts]

    def remove_stale_dbnode_files(self) -> None:
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
                for shard_id in range(self._num_shards):
                    file_name = osp.join(
                        self.path, self.name + "_" + str(shard_id) + file_ending
                    )
                    if osp.exists(file_name):
                        os.remove(file_name)

    def _get_cluster_conf_filename(self, port: int) -> str:
        """Returns the .conf file name for the given port number

        :param port: port number
        :type port: int
        :return: the dbnode configuration file name
        :rtype: str
        """
        return "".join(("nodes-", self.name, "-", str(port), ".conf"))

    def _get_cluster_conf_filenames(self, port: int) -> t.List[str]:  # cov-lsf
        """Returns the .conf file name for the given port number

        This function should bu used if and only if ``_mpmd==True``

        :param port: port number
        :type port: int
        :return: the dbnode configuration file name
        :rtype: str
        """
        return [
            "".join(("nodes-", self.name + f"_{shard_id}", "-", str(port), ".conf"))
            for shard_id in range(self._num_shards)
        ]

    @staticmethod
    def _parse_ips(filepath: str, num_ips: t.Optional[int] = None) -> t.List[str]:
        ips = []
        with open(filepath, "r", encoding="utf-8") as dbnode_file:
            lines = dbnode_file.readlines()
            for line in lines:
                content = line.split()
                if "IPADDRESS:" in content:
                    ips.append(content[-1])
                    if num_ips and len(ips) == num_ips:
                        break

        return ips

    def _parse_db_host(self, filepath: t.Optional[str] = None) -> str:
        """Parse the database host/IP from the output file

        If no file is passed as argument, then the first
        file in self._output_files is used.

        :param filepath: Path to file to parse
        :type filepath: str, optional
        :raises SmartSimError: if host/ip could not be found
        :return: ip address | hostname
        :rtype: str
        """
        if not filepath:
            filepath = osp.join(self.path, self._output_files[0])
        trials = 5
        ip_address = None

        # try a few times to give the database files time to
        # populate on busy systems.
        while not ip_address and trials > 0:
            try:
                if ip_addresses := self._parse_ips(filepath, 1):
                    ip_address = ip_addresses[0]
            # suppress error
            except FileNotFoundError:
                pass

            logger.debug("Waiting for Redis output files to populate...")
            if not ip_address:
                time.sleep(1)
                trials -= 1

        if not ip_address:
            logger.error(f"IP address lookup strategy failed for file {filepath}.")
            raise SmartSimError("Failed to obtain database hostname")

        return ip_address

    def _parse_db_hosts(self) -> t.List[str]:
        """Parse the database hosts/IPs from the output files

        this uses the RedisIP module that is built as a dependency
        The IP address is preferred, but if hostname is only present
        then a lookup to /etc/hosts is done through the socket library.
        This function must be called only if ``_mpmd==True``.

        :raises SmartSimError: if host/ip could not be found
        :return: ip addresses | hostnames
        :rtype: list[str]
        """
        ips: t.List[str] = []

        # Find out if all shards' output streams are piped to separate files
        if len(self._output_files) > 1:
            for output_file in self._output_files:
                filepath = osp.join(self.path, output_file)
                _ = self._parse_db_host(filepath)
        else:
            filepath = osp.join(self.path, self._output_files[0])
            trials = 10
            ips = []
            while len(ips) < self._num_shards and trials > 0:
                ips = []
                try:
                    ip_address = self._parse_ips(filepath, self._num_shards)
                    ips.extend(ip_address)

                # suppress error
                except FileNotFoundError:
                    pass

                if len(ips) < self._num_shards:
                    logger.debug("Waiting for RedisIP files to populate...")
                    # Larger sleep time, as this seems to be needed for
                    # multihost setups
                    time.sleep(2)
                    trials -= 1

            if len(ips) < self._num_shards:
                msg = f"IP address lookup strategy failed for file {filepath}. "
                msg += f"Found {len(ips)} out of {self._num_shards} IPs."
                logger.error(msg)
                raise SmartSimError("Failed to obtain database hostname")

        ips = list(dict.fromkeys(ips))
        return ips
