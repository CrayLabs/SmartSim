# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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

import fileinput
import itertools
import json
import os
import os.path as osp
import time
import typing as t
from dataclasses import dataclass

from .._core.config import CONFIG
from ..error import SmartSimError
from ..log import get_logger
from ..settings.base import RunSettings
from .entity import SmartSimEntity

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
        db_identifier: str = "",
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
        self.db_identifier = db_identifier

    @property
    def num_shards(self) -> int:
        if not hasattr(self.run_settings, "mpmd"):
            # return default number of shards if mpmd is not set
            return 1

        return len(self.run_settings.mpmd) + 1

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

    def clear_hosts(self) -> None:
        self._hosts = None

    @property
    def is_mpmd(self) -> bool:
        if not hasattr(self.run_settings, "mpmd"):
            # missing mpmd property guarantees this is not an mpmd run
            return False

        return bool(self.run_settings.mpmd)

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
    def _parse_launched_shard_info_from_iterable(
        stream: t.Iterable[str], num_shards: t.Optional[int] = None
    ) -> "t.List[LaunchedShardData]":
        lines = (line.strip() for line in stream)
        lines = (line for line in lines if line)
        tokenized = (line.split(maxsplit=1) for line in lines)
        tokenized = (tokens for tokens in tokenized if len(tokens) > 1)
        shard_data_jsons = (
            kwjson for first, kwjson in tokenized if "SMARTSIM_ORC_SHARD_INFO" in first
        )
        shard_data_kwargs = (json.loads(kwjson) for kwjson in shard_data_jsons)
        shard_data: "t.Iterable[LaunchedShardData]" = (
            LaunchedShardData(**kwargs) for kwargs in shard_data_kwargs
        )
        if num_shards:
            shard_data = itertools.islice(shard_data, num_shards)
        return list(shard_data)

    @classmethod
    def _parse_launched_shard_info_from_files(
        cls, file_paths: t.List[str], num_shards: t.Optional[int] = None
    ) -> "t.List[LaunchedShardData]":
        with fileinput.FileInput(file_paths) as ifstream:
            return cls._parse_launched_shard_info_from_iterable(ifstream, num_shards)

    def get_launched_shard_info(self) -> "t.List[LaunchedShardData]":
        """Parse the launched database shard info from the output files

        :raises SmartSimError: if all shard info could not be found
        :return: The found launched shard info
        :rtype: list[LaunchedShardData]
        """
        ips: "t.List[LaunchedShardData]" = []
        trials = CONFIG.database_file_parse_trials
        interval = CONFIG.database_file_parse_interval
        output_files = [osp.join(self.path, file) for file in self._output_files]

        while len(ips) < self.num_shards and trials > 0:
            try:
                ips = self._parse_launched_shard_info_from_files(
                    output_files, self.num_shards
                )
            except FileNotFoundError:
                ...
            if len(ips) < self.num_shards:
                logger.debug("Waiting for output files to populate...")
                time.sleep(interval)
                trials -= 1

        if len(ips) < self.num_shards:
            msg = (
                f"Failed to parse the launched DB shard information from file(s) "
                f"{', '.join(output_files)}. Found the information for "
                f"{len(ips)} out of {self.num_shards} DB shards."
            )
            logger.error(msg)
            raise SmartSimError(msg)
        return ips

    def _parse_db_hosts(self) -> t.List[str]:
        """Parse the database hosts/IPs from the output files

        The IP address is preferred, but if hostname is only present
        then a lookup to /etc/hosts is done through the socket library.

        :raises SmartSimError: if host/ip could not be found
        :return: ip addresses | hostnames
        :rtype: list[str]
        """
        return list({shard.hostname for shard in self.get_launched_shard_info()})


@dataclass(frozen=True)
class LaunchedShardData:
    """Data class to write and parse data about a launched database shard"""

    name: str
    hostname: str
    port: int
    cluster: bool

    @property
    def cluster_conf_file(self) -> t.Optional[str]:
        return f"nodes-{self.name}-{self.port}.conf" if self.cluster else None

    def to_dict(self) -> t.Dict[str, t.Any]:
        return dict(self.__dict__)
