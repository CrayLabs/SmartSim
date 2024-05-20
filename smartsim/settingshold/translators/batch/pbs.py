from __future__ import annotations

from copy import deepcopy
import typing as t
from ..batchArgTranslator import BatchArgTranslator
from ....error import SSConfigError
from ...common import IntegerArgument, StringArgument
from smartsim.log import get_logger                                                                                
from ...batchCommand import SchedulerType
logger = get_logger(__name__)

class QsubBatchArgTranslator(BatchArgTranslator):

    def scheduler_str(self) -> str:
        """ Get the string representation of the scheduler
        """
        return SchedulerType.PbsScheduler.value

    def set_nodes(self, num_nodes: int) -> t.Union[IntegerArgument,None]:
        """Set the number of nodes for this batch job

        In PBS, 'select' is the more primitive way of describing how
        many nodes to allocate for the job. 'nodes' is equivalent to
        'select' with a 'place' statement. Assuming that only advanced
        users would use 'set_resource' instead, defining the number of
        nodes here is sets the 'nodes' resource.

        :param num_nodes: number of nodes
        """

        return {"nodes": num_nodes}

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> t.Union[StringArgument,None]:
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
        return {"hostname": ",".join(host_list)}
    
    def set_walltime(self, walltime: str) -> t.Union[StringArgument,None]:
        """Set the walltime of the job

        format = "HH:MM:SS"

        If a walltime argument is provided in
        ``QsubBatchSettings.resources``, then
        this value will be overridden

        :param walltime: wall time
        """
        return {"walltime": walltime}
    
    def set_queue(self, queue: str) -> t.Union[StringArgument,None]:
        """Set the queue for the batch job

        :param queue: queue name
        """
        return {"q": str(queue)}
    
    def set_ncpus(self, num_cpus: int) -> t.Union[IntegerArgument,None]:
        """Set the number of cpus obtained in each node.

        If a select argument is provided in
        ``QsubBatchSettings.resources``, then
        this value will be overridden

        :param num_cpus: number of cpus per node in select
        """
        return {"ppn": int(num_cpus)}
    
    def set_account(self, account: str) -> t.Union[StringArgument,None]:
        """Set the account for this batch job

        :param acct: account id
        """
        return {"A": str(account)}

    def format_batch_args(self, batch_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Qsub
        :raises ValueError: if options are supplied without values
        """
        opts, batch_arg_copy = self._create_resource_list(batch_args)
        t.cast(t.List[str],batch_arg_copy)
        for opt, value in batch_arg_copy.items():
            prefix = "-"
            if not value:
                raise ValueError("PBS options without values are not allowed")
            opts += [" ".join((prefix + opt, str(value)))]
        return opts

    def _sanity_check_resources(
        self, batch_args: t.Dict[str, t.Union[str,int,float,None]]
    ) -> None:
        """Check that only select or nodes was specified in resources

        Note: For PBS Pro, nodes is equivalent to 'select' and 'place' so
        they are not quite synonyms. Here we assume that
        """
        checked_resources = batch_args

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
                    f"The type of {key} is {type(key)}. Only int and str "
                    "are allowed."
                )
            if not isinstance(value, (str, int)):
                raise TypeError(
                    f"The value associated with {key} is {type(value)}. Only int "
                    "and str are allowed."
                )

    def _create_resource_list(self, batch_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.Tuple[t.List[str],t.Dict[str, t.Union[str,int,float,None]]]:
        self._sanity_check_resources(batch_args)
        res = []

        batch_arg_copy = batch_args
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
            hosts_list = ["=".join(str(hosts))]
            t.cast(str,hosts_list)
            select_command += f":{'+'.join(hosts_list)}"
        res += [select_command]
        if walltime := batch_arg_copy.pop("walltime", None):
            res += [f"-l walltime={walltime}"]

        # # All other "standard" resource specs
        # for resource, value in batch_arg_copy.items():
        #     res += [f"-l {resource}={value}"]

        return res, batch_arg_copy