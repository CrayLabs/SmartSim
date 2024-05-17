from __future__ import annotations

from enum import Enum
import typing as t
from ..batchArgTranslator import BatchArgTranslator
from ...common import IntegerArgument, StringArgument
from smartsim.log import get_logger   
from ...batchCommand import SchedulerType                                                                             

logger = get_logger(__name__)

class BsubBatchArgTranslator(BatchArgTranslator):

    def scheduler_str(self) -> str:
        """ Get the string representation of the launcher
        """
        return SchedulerType.LsfScheduler.value

    def set_walltime(self, walltime: str) -> t.Union[StringArgument,None]:
        """Set the walltime

        This sets ``-W``.

        :param walltime: Time in hh:mm format, e.g. "10:00" for 10 hours,
                         if time is supplied in hh:mm:ss format, seconds
                         will be ignored and walltime will be set as ``hh:mm``
        """
        # For compatibility with other launchers, as explained in docstring
        if walltime:
            if len(walltime.split(":")) > 2:
                walltime = ":".join(walltime.split(":")[:2])
        return {"W": walltime}

    def set_smts(self, smts: int) -> t.Union[IntegerArgument,None]:
        """Set SMTs

        This sets ``-alloc_flags``. If the user sets
        SMT explicitly through ``-alloc_flags``, then that
        takes precedence.

        :param smts: SMT (e.g on Summit: 1, 2, or 4)
        """
        return {"alloc_flags": int(smts)}

    def set_project(self, project: str) -> t.Union[StringArgument,None]:
        """Set the project

        This sets ``-P``.

        :param time: project name
        """
        return {"P": project}
    
    def set_account(self, account: str) -> t.Union[StringArgument,None]:
        """Set the project

        this function is an alias for `set_project`.

        :param account: project name
        """
        return self.set_project(account)

    def set_nodes(self, num_nodes: int) -> t.Union[IntegerArgument,None]:
        """Set the number of nodes for this batch job

        This sets ``-nnodes``.

        :param nodes: number of nodes
        """
        return {"nnodes": int(num_nodes)}

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
            raise TypeError("host_list argument must be list of strings")
        return {"m": '"' + " ".join(host_list) + '"'}

    def set_tasks(self, tasks: int) -> t.Union[IntegerArgument,None]:
        """Set the number of tasks for this job

        This sets ``-n``

        :param tasks: number of tasks
        """
        return {"n": int(tasks)}

    def set_queue(self, queue: str) -> t.Union[StringArgument,None]:
        """Set the queue for this job

        :param queue: The queue to submit the job on
        """
        return {"q": queue}

    def format_batch_args(self, batch_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: list of batch arguments for Qsub
        """
        opts = []

        for opt, value in batch_args.items():

            prefix = "-"  # LSF only uses single dashses

            if not value:
                opts += [prefix + opt]
            else:
                opts += [" ".join((prefix + opt, str(value)))]

        return opts