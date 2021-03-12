from .settings import RunSettings, BatchSettings

class SrunSettings(RunSettings):
    def __init__(self, exe, exe_args=None, run_args=None, env_vars=None, alloc=None):
        """Initialize entity settings to run with Srun

        :param exe: executable
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: list[str] | str, optional
        :param run_args: srun arguments without dashes, defaults to None
        :type run_args: dict[str, str | None], optional
        :param env_vars: environment variables for job, defaults to None
        :type env_vars: dict[str, str], optional
        :param alloc: allocation ID if running on existing alloc, defaults to None
        :type alloc: str, optional
        """
        super().__init__(exe, exe_args, run_command="srun", run_args=run_args, env_vars=env_vars)
        self.alloc = alloc
        self.mpmd = False

    def set_nodes(self, num_nodes):
        """Set the number of nodes

        effectively this is setting: srun --nodes <num_nodes>

        :param num_nodes: number of nodes to run with
        :type num_nodes: int
        """
        self.run_args["nodes"] = int(num_nodes)

    def set_cpus_per_task(self, num_cpus):
        """Set the number of cpus to use per task

        This sets ``--cpus_per_task``

        :param num_cpus: number of cpus to use per task
        :type num_cpus: int
        """
        self.run_args["cpus-per-task"] = int(num_cpus)

    def set_tasks(self, num_tasks):
        """Set the number of tasks for this job

        This sets ``--ntasks``

        :param num_tasks: number of tasks
        :type num_tasks: int
        """
        self.run_args["ntasks"] = int(num_tasks)

    def set_tasks_per_node(self, num_tpn):
        """Set the number of tasks for this job

        This sets ``--ntasks-per-node``

        :param num_tpn: number of tasks per node
        :type num_tpn: int
        """
        self.run_args["ntasks-per-node"] = int(num_tpn)

    def format_run_args(self):
        """return a list of slurm formatted run arguments

        :return: list slurm arguments for these settings
        :rtype: list[str]
        """
        # add additional slurm arguments based on key length
        opts = []
        for opt, value in self.run_args.items():
            short_arg = bool(len(str(opt)) == 1)
            prefix = "-" if short_arg else "--"
            if not value:
                opts += [prefix + opt]
            else:
                if short_arg:
                    opts += [prefix + opt, str(value)]
                else:
                    opts += ["=".join((prefix + opt, str(value)))]
        return opts



class SbatchSettings(BatchSettings):
    def __init__(self, nodes=None, time="", account=None, batch_args=None):
        """Settings for a Sbatch workload

        Slurm Sbatch arguments can be written into batch_args
        as a dictionary. e.g. {'ntasks': 1}

        If the argument doesn't have a parameter, put None
        as the value. e.g. {'exclusive': None}

        Initialization values provided (nodes, time, account)
        will overwrite the same arguments in batch_args if present

        :param nodes: number of nodes, defaults to None
        :type nodes: int, optional
        :param time: walltime for job, e.g. "10:00:00" for 10 hours
        :type time: str, optional
        :param account: account for job, defaults to None
        :type account: str, optional
        :param batch_args: extra batch arguments, defaults to None
        :type batch_args: dict[str, str], optional
        """
        super().__init__("sbatch", batch_args=batch_args)
        if nodes:
            self.set_nodes(nodes)
        if time:
            self.set_walltime(time)
        if account:
            self.set_account(account)

    def set_walltime(self, walltime):
        """Set the walltime of the job

        format = "HH:MM:SS"

        :param walltime: wall time
        :type walltime: str
        """
        #TODO check for errors here
        self.batch_args["time"] = walltime

    def set_nodes(self, num_nodes):
        """Set the number of nodes for this batch job

        :param num_nodes: number of nodes
        :type num_nodes: int
        """
        self.batch_args["nodes"] = int(num_nodes)

    def set_account(self, acct):
        """Set the account for this batch job

        :param acct: account id
        :type acct: str
        """
        self.batch_args["account"] = acct

    def format_batch_args(self):
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Sbatch
        :rtype: list[str]
        """
        opts = []
        for opt, value in self.batch_args.items():
            # attach "-" prefix if argument is 1 character otherwise "--"
            short_arg = bool(len(str(opt)) == 1)
            prefix = "-" if short_arg else "--"

            if not value:
                opts += [prefix + opt]
            else:
                if short_arg:
                    opts += [prefix + opt, str(value)]
                else:
                    opts += ["=".join((prefix + opt, str(value)))]
        return opts
