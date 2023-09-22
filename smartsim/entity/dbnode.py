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

import itertools
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
        super().__init__(name, path, run_settings)
        self.ports = ports
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
        try:
            return len(self.run_settings.mpmd) + 1  # type: ignore[attr-defined]
        except AttributeError:
            return 1

    @property
    def host(self) -> str:
        try:
            (host,) = self.hosts
        except ValueError:
            raise ValueError(
                f"Multiple hosts detected for this DB Node: {', '.join(self.hosts)}"
            ) from None
        return host

    @property
    def hosts(self) -> t.List[str]:
        if not self._hosts:
            self._hosts = self._parse_db_hosts()
        return self._hosts

    @property
    def is_mpmd(self) -> bool:
        try:
            return bool(self.run_settings.mpmd)  # type: ignore[attr-defined]
        except AttributeError:
            return False

    def set_hosts(self, hosts: t.List[str]) -> None:
        self._hosts = [str(host) for host in hosts]

    def remove_stale_dbnode_files(self) -> None:
        """This function removes the .conf, .err, and .out files that
        have the same names used by this dbnode that may have been
        created from a previous experiment execution.
        """

        for port in self.ports:
            for conf_file in (
                osp.join(self.path, filename)
                for filename in self._get_cluster_conf_filenames(port)
            ):
                if osp.exists(conf_file):
                    os.remove(conf_file)

        for file_ending in [".err", ".out", ".mpmd"]:
            file_name = osp.join(self.path, self.name + file_ending)
            if osp.exists(file_name):
                os.remove(file_name)

        if self.is_mpmd:
            for file_ending in [".err", ".out"]:
                for shard_id in range(self.num_shards):
                    file_name = osp.join(
                        self.path, self.name + "_" + str(shard_id) + file_ending
                    )
                    if osp.exists(file_name):
                        os.remove(file_name)

    def _get_cluster_conf_filenames(self, port: int) -> t.List[str]:  # cov-lsf
        """Returns the .conf file name for the given port number

        This function should bu used if and only if ``_mpmd==True``

        :param port: port number
        :type port: int
        :return: the dbnode configuration file name
        :rtype: str
        """
        if self.num_shards == 1:
            return [f"nodes-{self.name}-{port}.conf"]
        return [
            f"nodes-{self.name}_{shard_id}-{port}.conf"
            for shard_id in range(self.num_shards)
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
        trials = 10
        output_files = [osp.join(self.path, file) for file in self._output_files]

        while len(ips) < self.num_shards and trials > 0:
            try:
                ips = list(
                    itertools.chain.from_iterable(
                        self._parse_ips(file) for file in output_files
                    )
                )
            except FileNotFoundError:
                ...

            if len(ips) < self.num_shards:
                logger.debug("Waiting for RedisIP files to populate...")
                # Larger sleep time, as this seems to be needed for
                # multihost setups
                time.sleep(2 if self.num_shards > 1 else 1)
                trials -= 1

        if len(ips) < self.num_shards:
            logger.error(
                f"IP address lookup strategy failed for file(s) "
                f"{', '.join(output_files)}. "
                f"Found {len(ips)} out of {self.num_shards} IPs."
            )
            raise SmartSimError("Failed to obtain database hostname")

        return list(set(ips))
