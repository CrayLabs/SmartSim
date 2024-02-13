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

import typing as t

from ..error import SSConfigError
from ..log import get_logger
from .base import BatchSettings

logger = get_logger(__name__)


class QsubBatchSettings(BatchSettings):
    def __init__(
        self,
        nodes: t.Optional[int] = None,
        ncpus: t.Optional[int] = None,
        time: t.Optional[str] = None,
        queue: t.Optional[str] = None,
        account: t.Optional[str] = None,
        resources: t.Optional[t.Dict[str, t.Union[str, int]]] = None,
        batch_args: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ):
        """Specify ``qsub`` batch parameters for a job

        ``nodes``, and ``ncpus`` are used to create the
        select statement for PBS if a select statement is not
        included in the ``resources``. If both are supplied
        the value for select statement supplied in ``resources``
        will override.

        :param nodes: number of nodes for batch, defaults to None
        :type nodes: int, optional
        :param ncpus: number of cpus per node, defaults to None
        :type ncpus: int, optional
        :param time: walltime for batch job, defaults to None
        :type time: str, optional
        :param queue: queue to run batch in, defaults to None
        :type queue: str, optional
        :param account: account for batch launch, defaults to None
        :type account: str, optional
        :param resources: overrides for resource arguments, defaults to None
        :type resources: dict[str, str], optional
        :param batch_args: overrides for PBS batch arguments, defaults to None
        :type batch_args: dict[str, str], optional
        """

        self._ncpus = ncpus

        self.resources = resources or {}
        resource_nodes = self.resources.get("nodes", None)

        if nodes and resource_nodes:
            raise ValueError(
                "nodes was incorrectly specified as a constructor parameter and also "
                "as a key in the resource mapping"
            )

        # time, queue, nodes, and account set in parent class init
        super().__init__(
            "qsub",
            batch_args=batch_args,
            nodes=nodes,
            account=account,
            queue=queue,
            time=time,
            **kwargs,
        )

        self._hosts: t.List[str] = []

    @property
    def resources(self) -> t.Dict[str, t.Union[str, int]]:
        return self._resources.copy()

    @resources.setter
    def resources(self, resources: t.Dict[str, t.Union[str, int]]) -> None:
        self._sanity_check_resources(resources)
        self._resources = resources.copy()

    def set_nodes(self, num_nodes: int) -> None:
        """Set the number of nodes for this batch job

        In PBS, 'select' is the more primitive way of describing how
        many nodes to allocate for the job. 'nodes' is equivalent to
        'select' with a 'place' statement. Assuming that only advanced
        users would use 'set_resource' instead, defining the number of
        nodes here is sets the 'nodes' resource.

        :param num_nodes: number of nodes
        :type num_nodes: int
        """

        if num_nodes:
            self.set_resource("nodes", num_nodes)

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :type host_list: str | list[str]
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be a list of strings")
        self._hosts = host_list

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        format = "HH:MM:SS"

        If a walltime argument is provided in
        ``QsubBatchSettings.resources``, then
        this value will be overridden

        :param walltime: wall time
        :type walltime: str
        """
        if walltime:
            self.set_resource("walltime", walltime)

    def set_queue(self, queue: str) -> None:
        """Set the queue for the batch job

        :param queue: queue name
        :type queue: str
        """
        if queue:
            self.batch_args["q"] = str(queue)

    def set_ncpus(self, num_cpus: t.Union[int, str]) -> None:
        """Set the number of cpus obtained in each node.

        If a select argument is provided in
        ``QsubBatchSettings.resources``, then
        this value will be overridden

        :param num_cpus: number of cpus per node in select
        :type num_cpus: int
        """
        self._ncpus = int(num_cpus)

    def set_account(self, account: str) -> None:
        """Set the account for this batch job

        :param acct: account id
        :type acct: str
        """
        if account:
            self.batch_args["A"] = str(account)

    def set_resource(self, resource_name: str, value: t.Union[str, int]) -> None:
        """Set a resource value for the Qsub batch

        If a select statement is provided, the nodes and ncpus
        arguments will be overridden. Likewise for Walltime

        :param resource_name: name of resource, e.g. walltime
        :type resource_name: str
        :param value: value
        :type value: str
        """
        # TODO add error checking here
        # TODO include option to overwrite place (warning for orchestrator?)
        updated_dict = self.resources
        updated_dict.update({resource_name: value})
        self._sanity_check_resources(updated_dict)
        self.resources = updated_dict

    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Qsub
        :rtype: list[str]
        :raises ValueError: if options are supplied without values
        """
        opts = self._create_resource_list()
        for opt, value in self.batch_args.items():
            prefix = "-"
            if not value:
                raise ValueError("PBS options without values are not allowed")
            opts += [" ".join((prefix + opt, str(value)))]
        return opts

    def _sanity_check_resources(
        self, resources: t.Optional[t.Dict[str, t.Union[str, int]]] = None
    ) -> None:
        """Check that only select or nodes was specified in resources

        Note: For PBS Pro, nodes is equivalent to 'select' and 'place' so
        they are not quite synonyms. Here we assume that
        """
        # Note: isinstance check here to avoid collision with default
        checked_resources = resources if isinstance(resources, dict) else self.resources

        has_select = checked_resources.get("select", None)
        has_nodes = checked_resources.get("nodes", None)

        if has_select and has_nodes:
            raise SSConfigError(
                "'select' and 'nodes' cannot both be specified. This can happen "
                "if nodes were specified using the 'set_nodes' method and "
                "'select' was set using 'set_resource'. Please only specify one."
            )

        if has_select and not isinstance(has_select, int):
            raise TypeError("The value for 'select' must be an integer")
        if has_nodes and not isinstance(has_nodes, int):
            raise TypeError("The value for 'nodes' must be an integer")

        for key, value in checked_resources.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"The type of {key=} is {type(key)}. Only int and str "
                    "are allowed."
                )
            if not isinstance(value, (str, int)):
                raise TypeError(
                    f"The value associated with {key=} is {type(value)}. Only int "
                    "and str are allowed."
                )

    def _create_resource_list(self) -> t.List[str]:
        self._sanity_check_resources()
        res = []

        # Pop off some specific keywords that need to be treated separately
        resources = self.resources  # Note this is a copy so not modifying original

        # Construct the basic select/nodes statement
        if select := resources.pop("select", None):
            select_command = f"-l select={select}"
        elif nodes := resources.pop("nodes", None):
            select_command = f"-l nodes={nodes}"
        else:
            raise SSConfigError(
                "Insufficient resource specification: no nodes or select statement"
            )
        if self._ncpus:
            select_command += f":ncpus={self._ncpus}"
        if self._hosts:
            hosts = ["=".join(("host", str(host))) for host in self._hosts]
            select_command += f":{'+'.join(hosts)}"
        res += [select_command]

        # All other "standard" resource specs
        for resource, value in resources.items():
            res += [f"-l {resource}={value}"]

        return res
