
from smartsim.error.errors import SmartSimError
from .settings import BatchSettings
from ..error import SSConfigError
from ..utils.helpers import init_default

class QsubBatchSettings(BatchSettings):
    def __init__(self, nodes=None, ncpus=None, time=None, account=None, resources=None, batch_args=None, **kwargs):
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
        if account:
            self.set_account(account)

    def set_nodes(self, num_nodes):
        """Set the number of nodes for this batch job

        If a select argument is provided in QsubBatchSEttings.resources
        this value will be overridden

        :param num_nodes: number of nodes
        :type num_nodes: int
        """
        self._nodes = int(num_nodes)

    def set_walltime(self, walltime):
        """Set the walltime of the job

        format = "HH:MM:SS"

        If a walltime argument is provided in QsubBatchSEttings.resources
        this value will be overridden

        :param walltime: wall time
        :type walltime: str
        """
        self._time = walltime

    def set_ncpus(self, num_cpus):
        """Set the number of cpus obtained in each node.

        If a select argument is provided in QsubBatchSEttings.resources
        this value will be overridden

        :param num_cpus: [description]
        :type num_cpus: [type]
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
        #TODO add error checking here
        #TODO include option to overwrite place (warning for orchestrator?)
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
            if self._nodes and self._ncpus:
                res += [f"-l select={str(self._nodes)}:ncpus={str(self._ncpus)}"]
            elif self._nodes:
                res += [f"-l select={str(self._nodes)}"]
            else:
                raise SmartSimError(
                    "Insufficient resource specification: no nodes or select statement")

        # TODO open user option path for placement
        res += ["-l place=scatter"]

        # get time from resources or kwargs
        if "walltime" in self.resources:
            res += [f"-l walltime={str(self.resources['walltime'])}"]
        else:
            if self._time:
                res += [f"-l walltime={self._time}"]

        for resource, value in self.resources.items():
            if resource not in ["select", "walltime"]:
                res += [f"-l {resource}={str(value)}"]
        return res

