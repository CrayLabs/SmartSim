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

from ..error import LauncherUnsupportedFeature, SSConfigError
from ..log import get_logger
from .base import BatchSettings

logger = get_logger(__name__)


class SgeQsubBatchSettings(BatchSettings):
    def __init__(
        self,
        time: t.Optional[str] = None,
        ncpus: t.Optional[int] = None,
        pe_type: t.Optional[str] = None,
        account: t.Optional[str] = None,
        shebang: str = "#!/bin/bash -l",
        resources: t.Optional[t.Dict[str, t.Union[str, int]]] = None,
        batch_args: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ):
        """Specify SGE batch parameters for a job

        :param time: walltime for batch job
        :param ncpus: number of cpus per node
        :param pe_type: type of parallel environment
        :param queue: queue to run batch in
        :param account: account for batch launch
        :param resources: overrides for resource arguments
        :param batch_args: overrides for SGE batch arguments
        """

        if "nodes" in kwargs:
            kwargs["nodes"] = 0

        self.resources = resources or {}
        if ncpus:
            self.set_ncpus(ncpus)
        if pe_type:
            self.set_pe_type(pe_type)
        self.set_shebang(shebang)

        # time, queue, nodes, and account set in parent class init
        super().__init__(
            "qsub",
            batch_args=batch_args,
            account=account,
            time=time,
            **kwargs,
        )

        self._context_variables: t.List[str] = []
        self._env_vars: t.Dict[str, str] = {}

    @property
    def resources(self) -> t.Dict[str, t.Union[str, int]]:
        return self._resources.copy()

    @resources.setter
    def resources(self, resources: t.Dict[str, t.Union[str, int]]) -> None:
        self._sanity_check_resources(resources)
        self._resources = resources.copy()

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        raise LauncherUnsupportedFeature(
            "SGE does not support requesting specific hosts in batch jobs"
        )

    def set_queue(self, queue: str) -> None:
        raise LauncherUnsupportedFeature("SGE does not support specifying queues")

    def set_shebang(self, shebang: str) -> None:
        """Set the shebang (shell) for the batch job

        :param shebang: The shebang used to interpret the rest of script
                        (e.g. #!/bin/bash)
        """
        self.shebang = shebang

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        format = "HH:MM:SS"

        If a walltime argument is provided in
        ``SGEBatchSettings.resources``, then
        this value will be overridden

        :param walltime: wall time
        """
        if walltime:
            self.set_resource("h_rt", walltime)

    def set_nodes(self, num_nodes: t.Optional[int]) -> None:
        """Set the number of nodes, invalid for SGE

        :param nodes: Number of nodes, any integer other than 0 is invalid
        """
        if num_nodes:
            raise LauncherUnsupportedFeature(
                "SGE does not support setting the number of nodes"
            )

    def set_ncpus(self, num_cpus: t.Union[int, str]) -> None:
        """Set the number of cpus obtained in each node.

        :param num_cpus: number of cpus per node in select
        """
        self.set_resource("ncpus", int(num_cpus))

    def set_ngpus(self, num_gpus: t.Union[int, str]) -> None:
        """Set the number of GPUs obtained in each node.

        :param num_gpus: number of GPUs per node in select
        """
        self.set_resource("gpu", num_gpus)

    def set_account(self, account: str) -> None:
        """Set the account for this batch job

        :param acct: account id
        """
        if account:
            self.batch_args["A"] = str(account)

    def set_project(self, project: str) -> None:
        """Set the project for this batch job

        :param acct: project id
        """
        if project:
            self.batch_args["P"] = str(project)

    def update_context_variables(
        self,
        action: t.Literal["ac", "sc", "dc"],
        var_name: str,
        value: t.Optional[t.Union[int, str]] = None,
    ) -> None:
        """
        Add, set, or delete context variables

        Configure any context variables using SGE's -ac, -sc, and -dc
        qsub switches. These modifications are appended each time this
        method is called, so the order does matter

        :param action: Add, set, or delete a context variable (ac, dc, or sc)
        :param var_name: The name of the variable to set
        :param value: The value of the variable
        """
        if action not in ["ac", "sc", "dc"]:
            raise ValueError("The action argument must be ac, sc, or dc")
        if action == "dc" and value:
            raise SSConfigError("When using the 'dc' action, value should not be set")

        command = f"-{action} {var_name}"
        if value:
            command += f"={value}"
        self._context_variables.append(command)

    def set_hyperthreading(self, enable: bool = True) -> None:
        """Enable or disable hyperthreading

        :param enable: Enable (True) or disable (False) hypthreading
        """
        self.set_resource("threads", int(enable))

    def set_memory_per_pe(self, memory_spec: str) -> None:
        """Set the amount of memory per processing element

        :param memory_spec: The amount of memory per PE (e.g. 2G)
        """
        self.set_resource("mem", memory_spec)

    def set_pe_type(self, pe_type: str) -> None:
        """Set the parallel environment

        :param pe_type: parallel environment identifier (e.g. mpi or smp)
        """
        if pe_type:
            self.set_resource("pe_type", pe_type)

    def set_threads_per_pe(self, threads_per_core: int) -> None:
        """Sets the number of threads per processing element

        :param threads_per_core: Number of threads per core
        """

        self._env_vars["OMP_NUM_THREADS"] = str(threads_per_core)

    def set_resource(self, resource_name: str, value: t.Union[str, int]) -> None:
        """Set a resource value for the SGE batch

        If a select statement is provided, the nodes and ncpus
        arguments will be overridden. Likewise for Walltime

        :param resource_name: name of resource, e.g. walltime
        :param value: value
        """
        updated_dict = self.resources
        updated_dict.update({resource_name: value})
        self._sanity_check_resources(updated_dict)
        self.resources = updated_dict

    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: batch arguments for SGE
        :raises ValueError: if options are supplied without values
        """
        opts = self._create_resource_list()
        for opt, value in self.batch_args.items():
            prefix = "-"
            if not value:
                raise ValueError("SGE options without values are not allowed")
            opts += [" ".join((prefix + opt, str(value)))]
        return opts

    def _sanity_check_resources(
        self, resources: t.Optional[t.Dict[str, t.Union[str, int]]] = None
    ) -> None:
        """Check that resources are correctly formatted"""
        # Note: isinstance check here to avoid collision with default
        checked_resources = resources if isinstance(resources, dict) else self.resources

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

        # Construct the configuration of the parallel environment
        ncpus = resources.pop("ncpus", None)
        pe_type = resources.pop("pe_type", None)
        if (pe_type is None and ncpus) or (pe_type and ncpus is None):
            msg = f"{ncpus=} and {pe_type=} must both be set. "
            msg += "Call set_ncpus and/or set_pe_type."
            raise SSConfigError(msg)

        if pe_type and ncpus:
            res += [f"-pe {pe_type} {ncpus}"]

        # Deal with context variables
        for context_variable in self._context_variables:
            res += [context_variable]

        # All other "standard" resource specs
        for resource, value in resources.items():
            res += [f"-l {resource}={value}"]

        # Set any environment variables
        for key, value in self._env_vars.items():
            res += [f"-v {key}={value}"]
        return res
