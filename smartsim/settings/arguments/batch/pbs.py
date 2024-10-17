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

from __future__ import annotations

import typing as t
from copy import deepcopy

from smartsim.log import get_logger

from ....error import SSConfigError
from ...batch_command import BatchSchedulerType
from ...common import StringArgument
from ..batch_arguments import BatchArguments

logger = get_logger(__name__)


class QsubBatchArguments(BatchArguments):
    """A class to represent the arguments required for submitting batch
    jobs using the qsub command.
    """

    def scheduler_str(self) -> str:
        """Get the string representation of the scheduler

        :returns: The string representation of the scheduler
        """
        return BatchSchedulerType.Pbs.value

    def set_nodes(self, num_nodes: int) -> None:
        """Set the number of nodes for this batch job

        In PBS, 'select' is the more primitive way of describing how
        many nodes to allocate for the job. 'nodes' is equivalent to
        'select' with a 'place' statement. Assuming that only advanced
        users would use 'set_resource' instead, defining the number of
        nodes here is sets the 'nodes' resource.

        :param num_nodes: number of nodes
        """

        self.set("nodes", str(num_nodes))

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be a list of strings")
        self.set("hostname", ",".join(host_list))

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        format = "HH:MM:SS"

        If a walltime argument is provided in
        ``QsubBatchSettings.resources``, then
        this value will be overridden

        :param walltime: wall time
        """
        self.set("walltime", walltime)

    def set_queue(self, queue: str) -> None:
        """Set the queue for the batch job

        :param queue: queue name
        """
        self.set("q", str(queue))

    def set_ncpus(self, num_cpus: int) -> None:
        """Set the number of cpus obtained in each node.

        If a select argument is provided in
        ``QsubBatchSettings.resources``, then
        this value will be overridden

        :param num_cpus: number of cpus per node in select
        """
        self.set("ppn", str(num_cpus))

    def set_account(self, account: str) -> None:
        """Set the account for this batch job

        :param acct: account id
        """
        self.set("A", str(account))

    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: batch arguments for `qsub`
        :raises ValueError: if options are supplied without values
        """
        opts, batch_arg_copy = self._create_resource_list(self._batch_args)
        for opt, value in batch_arg_copy.items():
            prefix = "-"
            if not value:
                raise ValueError("PBS options without values are not allowed")
            opts += [f"{prefix}{opt}", str(value)]
        return opts

    @staticmethod
    def _sanity_check_resources(batch_args: t.Dict[str, str | None]) -> None:
        """Check that only select or nodes was specified in resources

        Note: For PBS Pro, nodes is equivalent to 'select' and 'place' so
        they are not quite synonyms. Here we assume that
        """

        has_select = batch_args.get("select", None)
        has_nodes = batch_args.get("nodes", None)

        if has_select and has_nodes:
            raise SSConfigError(
                "'select' and 'nodes' cannot both be specified. This can happen "
                "if nodes were specified using the 'set_nodes' method and "
                "'select' was set using 'set_resource'. Please only specify one."
            )

    def _create_resource_list(
        self, batch_args: t.Dict[str, str | None]
    ) -> t.Tuple[t.List[str], t.Dict[str, str | None]]:
        self._sanity_check_resources(batch_args)
        res = []

        batch_arg_copy = deepcopy(batch_args)
        # Construct the basic select/nodes statement
        if select := batch_arg_copy.pop("select", None):
            select_command = f"-l select={select}"
        elif nodes := batch_arg_copy.pop("nodes", None):
            select_command = f"-l nodes={nodes}"
        else:
            raise SSConfigError(
                "Insufficient resource specification: no nodes or select statement"
            )
        if ncpus := batch_arg_copy.pop("ppn", None):
            select_command += f":ncpus={ncpus}"
        if hosts := batch_arg_copy.pop("hostname", None):
            hosts_list = ["=".join(("host", str(host))) for host in hosts.split(",")]
            select_command += f":{'+'.join(hosts_list)}"
        res += select_command.split()
        if walltime := batch_arg_copy.pop("walltime", None):
            res += ["-l", f"walltime={walltime}"]

        return res, batch_arg_copy

    def set(self, key: str, value: str | None) -> None:
        """Set an arbitrary launch argument

        :param key: The launch argument
        :param value: A string representation of the value for the launch
            argument (if applicable), otherwise `None`
        """
        self._batch_args[key] = value
