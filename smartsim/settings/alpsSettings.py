from .settings import RunSettings

class AprunSettings(RunSettings):
    def __init__(self, exe, exe_args=None, run_args=None, env_vars=None):
        """Settings to run job with ``aprun`` command

        :param exe: executable
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: str | list[str], optional
        :param run_args: arguments for run command, defaults to None
        :type run_args: dict[str, str], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(exe, exe_args, run_command="aprun", run_args=run_args, env_vars=env_vars)
        self.mpmd = []

    def make_mpmd(self, aprun_settings):
        self.mpmd.append(aprun_settings)

    def set_cpus_per_task(self, num_cpus):
        """Set the number of cpus to use per task

        This sets ``--cpus-per-pe``

        :param num_cpus: number of cpus to use per task
        :type num_cpus: int
        """
        self.run_args["cpus-per-pe"] = int(num_cpus)

    def set_tasks(self, num_tasks):
        """Set the number of tasks for this job

        This sets ``--pes``

        :param num_tasks: number of tasks
        :type num_tasks: int
        """
        self.run_args["pes"] = int(num_tasks)

    def set_tasks_per_node(self, num_tpn):
        """Set the number of tasks for this job

        This sets ``--pes-per-node``

        :param num_tpn: number of tasks per node
        :type num_tpn: int
        """
        self.run_args["pes-per-node"] = int(num_tpn)

    def format_run_args(self):
        """return a list of PBSPro formatted run arguments

        :return: list PBSPro arguments for these settings
        :rtype: list[str]
        """
        # args launcher uses
        args = []
        restricted = ["wdir"]

        for opt, value in self.run_args.items():
            if opt not in restricted:
                short_arg = bool(len(str(opt)) == 1)
                prefix = "-" if short_arg else "--"
                if not value:
                    args += [prefix + opt]
                else:
                    if short_arg:
                        args += [prefix + opt, str(value)]
                    else:
                        args += ["=".join((prefix + opt, str(value)))]
        return args

    def format_env_vars(self):
        """Format the environment variables for aprun

        :return: list of env vars
        :rtype: list[str]
        """
        formatted = []
        if self.env_vars:
            for name, value in self.env_vars.items():
                formatted += ["-e", name + "=" + str(value)]
        return formatted
