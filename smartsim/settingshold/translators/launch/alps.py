from ..launchArgTranslator import LaunchArgTranslator
import typing as t
from ...common import IntegerArgument, StringArgument
from ...launchCommand import LauncherType
from smartsim.log import get_logger                                                                                

logger = get_logger(__name__)

class AprunArgTranslator(LaunchArgTranslator):

    def launcher_str(self) -> str:
        """ Get the string representation of the launcher
        """
        return LauncherType.AlpsLauncher.value
    
    def set_cpus_per_task(self, cpus_per_task: int) -> t.Union[IntegerArgument, None]:
        """Set the number of cpus to use per task

        This sets ``--cpus-per-pe``

        :param cpus_per_task: number of cpus to use per task
        """
        return {"cpus-per-pe": int(cpus_per_task)}

    def set_tasks(self, tasks: int) -> t.Union[IntegerArgument,None]:
        """Set the number of tasks for this job

        This sets ``--pes``

        :param tasks: number of tasks
        """
        return {"pes": int(tasks)}

    def set_tasks_per_node(self, tasks_per_node: int) -> t.Union[IntegerArgument, None]:
        """Set the number of tasks for this job

        This sets ``--pes-per-node``

        :param tasks_per_node: number of tasks per node
        """
        return {"pes-per-node": int(tasks_per_node)}

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> t.Union[StringArgument, None]:
        """Specify the hostlist for this job

        This sets ``--node-list``
        
        :param host_list: hosts to launch on
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        return {"node-list": ",".join(host_list)}

    def set_hostlist_from_file(self, file_path: str) -> t.Union[StringArgument, None]:
        """Use the contents of a file to set the node list

        This sets ``--node-list-file``

        :param file_path: Path to the hostlist file
        """
        return {"node-list-file": file_path}
    
    def set_excluded_hosts(self, host_list: t.Union[str, t.List[str]]) -> t.Union[StringArgument, None]:
        """Specify a list of hosts to exclude for launching this job
        
        This sets ``--exclude-node-list``

        :param host_list: hosts to exclude
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        return {"exclude-node-list": ",".join(host_list)}

    def set_cpu_bindings(self, bindings: t.Union[int, t.List[int]]) -> t.Union[StringArgument, None]:
        """Specifies the cores to which MPI processes are bound

        This sets ``--cpu-binding``

        :param bindings: List of cpu numbers
        """
        if isinstance(bindings, int):
            bindings = [bindings]
        return {"cpu-binding": ",".join(str(int(num)) for num in bindings)}

    def set_memory_per_node(self, memory_per_node: int) -> t.Union[StringArgument, None]:
        """Specify the real memory required per node

        This sets ``--memory-per-pe`` in megabytes

        :param memory_per_node: Per PE memory limit in megabytes
        """
        return {"memory-per-pe": str(memory_per_node)}

    def set_walltime(self, walltime: str) -> t.Union[StringArgument, None]:
        """Set the walltime of the job

        Walltime is given in total number of seconds

        :param walltime: wall time
        """
        return {"cpu-time-limit": str(walltime)}

    def set_verbose_launch(self, verbose: bool) -> t.Union[t.Dict[str, None], t.Dict[str, int], None]:
        """Set the job to run in verbose mode

        This sets ``--debug`` arg to the highest level

        :param verbose: Whether the job should be run verbosely
        """
        return {"debug": 7}

    def set_quiet_launch(self, quiet: bool) -> t.Union[t.Dict[str,None],None]:
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        """
        return {"quiet": None}

    def format_env_vars(self, env_vars: t.Optional[t.Dict[str, t.Optional[str]]]) -> t.Union[t.List[str],None]:
        """Format the environment variables for aprun

        :return: list of env vars
        """
        formatted = []
        if env_vars:
            for name, value in env_vars.items():
                formatted += ["-e", name + "=" + str(value)]
        return formatted

    def format_launcher_args(self, launch_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.Union[t.List[str],None]:
        """Return a list of ALPS formatted run arguments

        :return: list of ALPS arguments for these settings
        """
        # args launcher uses
        args = []
        restricted = ["wdir"]

        for opt, value in launch_args.items():
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