from smartsim.error.errors import SmartSimError

from ..error import SSConfigError
from ..utils.helpers import init_default
from .settings import BatchSettings


class CobaltBatchSettings(BatchSettings):
    def __init__(self, nodes=None, time="", queue=None, account=None, batch_args=None):
        """Specify settings for a Cobalt ``qsub`` batch launch

        If the argument doesn't have a parameter, put None
        as the value. e.g. {'exclusive': None}

        Initialization values provided (nodes, time, account)
        will overwrite the same arguments in ``batch_args`` if present

        :param nodes: number of nodes
        :type nodes: int, optional
        :param time: walltime for job, e.g. "10:00:00" for 10 hours
        :type time: str, optional
        :param queue: queue to launch job in
        :type queue: str
        :param account: account for job
        :type account: str, optional
        :param batch_args: extra batch arguments
        :type batch_args: dict[str, str], optional
        """
        super().__init__("qsub", batch_args=batch_args)
        if nodes:
            self.set_nodes(nodes)
        if time:
            self.set_walltime(time)
        if account:
            self.set_account(account)
        if queue:
            self.set_queue(queue)

    def set_walltime(self, walltime):
        """Set the walltime of the job

        format = "HH:MM:SS"

        Cobalt walltime can also be specified with number
        of minutes.

        :param walltime: wall time
        :type walltime: str
        """
        # TODO check for formatting errors here
        # TODO catch existing "t" in batch_args
        self.batch_args["time"] = walltime

    def set_nodes(self, num_nodes):
        """Set the number of nodes for this batch job

        :param num_nodes: number of nodes
        :type num_nodes: int
        """
        # TODO catch existing "n" in batch_args
        self.batch_args["nodecount"] = int(num_nodes)

    def set_hostlist(self, host_list):
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :type host_list: list[str]
        :raises TypeError:
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all([isinstance(host, str) for host in host_list]):
            raise TypeError("host_list argument must be list of strings")
        hosts = ",".join(host_list)
        self.batch_args["attrs"] = f"location={hosts}"

    def set_tasks(self, num_tasks):
        """Set total number of processes to start

        :param num_tasks: number of processes
        :type num_tasks: int
        """
        self.batch_args["proccount"] = int(num_tasks)

    def set_queue(self, queue):
        """Set the queue for the batch job

        :param queue: queue name
        :type queue: str
        """
        # TODO catch existing "q" in batch args
        self.batch_args["queue"] = str(queue)

    def set_account(self, acct):
        """Set the account for this batch job

        :param acct: account id
        :type acct: str
        """
        # TODO catch existing "A" in batch_args
        self.batch_args["project"] = acct

    def format_batch_args(self):
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Sbatch
        :rtype: list[str]
        """
        restricted = [
            "o",
            "output",  # output is determined by interface
            "O",
            "outputprefix",  # step name is output prefix
            "e",
            "error",  # error is determined by interface
            "cwd",  # cwd is determined by interface
            "jobname",  # step name is jobname
        ]
        opts = []
        for opt, value in self.batch_args.items():
            if opt not in restricted:
                # attach "-" prefix if argument is 1 character otherwise "--"
                short_arg = bool(len(str(opt)) == 1)
                prefix = "-" if short_arg else "--"

                if not value:
                    opts += [prefix + opt]
                else:
                    if short_arg:
                        opts += [prefix + opt, str(value)]
                    else:
                        opts += [" ".join((prefix + opt, str(value)))]
        return opts
