from smartsim.error.errors import SmartSimError

from ..error import SSConfigError
from ..utils.helpers import init_default
from .settings import BatchSettings


class QsubBatchSettings(BatchSettings):
    def __init__(
        self,
        nodes=None,
        ncpus=None,
        time=None,
        queue=None,
        account=None,
        resources=None,
        batch_args=None,
        **kwargs,
    ):
        """Create a Qsub batch setting for an entity

        :param nodes: number of nodes for batch, defaults to None
        :type nodes: int, optional
        :param ncpus: number of cpus per node, defaults to None
        :type ncpus: int, optional
        :param time: walltime, defaults to None
        :type time: str, optional
        :param account: account for batch launch, defaults to None
        :type account: str, optional
        :param resources: overrides for resource arguments, defaults to None
        :type resources: dict[str, str], optional
        :param batch_args: overrides for PBS batch arguments, defaults to None
        :type batch_args: dict[str, str], optional
        """
        super().__init__("qsub", batch_args=batch_args)
        self.resources = init_default({}, resources, dict)
        self._nodes = nodes
        self._time = time
        self._ncpus = ncpus
        self._hosts = None
        if account:
            self.set_account(account)
        if queue:
            self.set_queue(queue)

    def set_nodes(self, num_nodes):
        """Set the number of nodes for this batch job

        If a select argument is provided in QsubBatchSEttings.resources
        this value will be overridden

        :param num_nodes: number of nodes
        :type num_nodes: int
        """
        self._nodes = int(num_nodes)

    def set_hostlist(self, host_list):
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all([isinstance(host, str) for host in host_list]):
            raise TypeError("host_list argument must be list of strings")
        self._hosts = host_list

    def set_walltime(self, walltime):
        """Set the walltime of the job

        format = "HH:MM:SS"

        If a walltime argument is provided in QsubBatchSEttings.resources
        this value will be overridden

        :param walltime: wall time
        :type walltime: str
        """
        self._time = walltime

    def set_queue(self, queue):
        """Set the queue for the batch job

        :param queue: queue name
        :type queue: str
        """
        self.batch_args["q"] = str(queue)

    def set_ncpus(self, num_cpus):
        """Set the number of cpus obtained in each node.

        If a select argument is provided in QsubBatchSettings.resources
        this value will be overridden

        :param num_cpus: number of cpus per node in select
        :type num_cpus: int
        """
        self._ncpus = int(num_cpus)

    def set_account(self, acct):
        """Set the account for this batch job

        :param acct: account id
        :type acct: str
        """
        self.batch_args["A"] = str(acct)

    def set_resource(self, resource_name, value):
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
        self.resources[resource_name] = value

    def format_batch_args(self):
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Qsub
        :rtype: list[str]
        """
        opts = self._create_resource_list()
        for opt, value in self.batch_args.items():
            prefix = "-"
            if not value:
                raise SSConfigError("PBS options without values are not allowed")
            opts += [" ".join((prefix + opt, str(value)))]
        return opts

    def _create_resource_list(self):
        res = []

        # get select statement from resources or kwargs
        if "select" in self.resources:
            res += [f"-l select={str(self.resources['select'])}"]
        else:
            select = "-l select="
            if self._nodes:
                select += str(self._nodes)
            else:
                raise SmartSimError(
                    "Insufficient resource specification: no nodes or select statement"
                )
            if self._ncpus:
                select += f":ncpus={self._ncpus}"
            if self._hosts:
                hosts = ["=".join(("host", str(host))) for host in self._hosts]
                select += f":{'+'.join(hosts)}"
            res += [select]

        if "place" in self.resources:
            res += [f"-l place={str(self.resources['place'])}"]
        else:
            res += ["-l place=scatter"]

        # get time from resources or kwargs
        if "walltime" in self.resources:
            res += [f"-l walltime={str(self.resources['walltime'])}"]
        else:
            if self._time:
                res += [f"-l walltime={self._time}"]

        for resource, value in self.resources.items():
            if resource not in ["select", "walltime", "place"]:
                res += [f"-l {resource}={str(value)}"]
        return res
