from __future__ import annotations

from abc import ABC, abstractmethod
import typing as t
# from launchcommand import LauncherType

from smartsim.log import get_logger                                                                                    

logger = get_logger(__name__)

from ..common import IntegerArgument, StringArgument, FloatArgument   


class LaunchArgTranslator(ABC):
    """Abstract base class that defines all generic launcher
    argument methods that are not supported.  It is the
    responsibility of child classes for each launcher to translate
    the input parameter to a properly formatted launcher argument.
    """

    @abstractmethod
    def launcher_str(self) -> str:
        """ Get the string representation of the launcher
        """
        pass

    def set_nodes(self, nodes: int) -> t.Union[IntegerArgument, None]:
        """ Convert the provide number of nodes into a properly formatted launcher 
        argument.
        """
        logger.warning(f"set_nodes() not supported for {self.launcher_str()}.")
        return None
    
    def set_hostlist(self, hostlist: t.Union[str, t.List[str]]) -> t.Union[StringArgument,None]:
        """ Convert the provide hostlist into a properly formatted launcher argument.
        """
        logger.warning(f"set_hostlist() not supported for {self.launcher_str()}.")
        return None
    
    def set_hostlist_from_file(self, hostlist: str) -> t.Union[StringArgument,None]:
        """ Convert the file path into a properly formatted launcher argument.
        """
        logger.warning(f"set_hostlist_from_file() not supported for {self.launcher_str()}.")
        return None
    
    def set_excluded_hosts(self, hostlist: t.Union[str, t.List[str]]) -> t.Union[StringArgument,None]:
        """ Convert the hostlist into a properly formatted launcher argument.
        """
        logger.warning(f"set_excluded_hosts() not supported for {self.launcher_str()}.")
        return None
    
    def set_cpus_per_task(self, cpus_per_task: int) -> t.Union[IntegerArgument,None]:
        """ Convert the cpus_per_task into a properly formatted launcher argument.
        """
        logger.warning(f"set_cpus_per_task() not supported for {self.launcher_str()}.")
        return None
    
    def set_tasks(self, tasks: int) -> t.Union[IntegerArgument,None]:
        """ Convert the tasks into a properly formatted launcher argument.
        """
        logger.warning(f"set_tasks() not supported for {self.launcher_str()}.")
        return None

    def set_tasks_per_node(self, tasks: int) -> t.Union[IntegerArgument,None]:
        """ Convert the set_tasks_per_node into a properly formatted launcher argument.
        """
        logger.warning(f"set_tasks_per_node() not supported for {self.launcher_str()}.")
        return None

    def set_cpu_bindings(self, bindings: t.Union[int,t.List[int]]) -> t.Union[StringArgument,None]:
        """ Convert the cpu bindings into a properly formatted launcher argument.
        """
        logger.warning(f"set_cpu_bindings() not supported for {self.launcher_str()}.")
        return None

    def set_memory_per_node(self, memory_per_node: int) -> t.Union[StringArgument,None]:
        """ Convert the real memory required per node into a properly formatted
        launcher argument.
        """
        logger.warning(f"set_memory_per_node() not supported for {self.launcher_str()}.")
        return None

    def set_executable_broadcast(self, dest_path: str) -> t.Union[StringArgument,None]:
        """ Convert executable file to be copied to allocated compute nodes into
        a properly formatted launcher argument.
        """
        logger.warning(f"set_executable_broadcast() not supported for {self.launcher_str()}.")
        return None

    def set_node_feature(self, feature_list: t.Union[str, t.List[str]]) -> t.Union[StringArgument,None]:
        """ Convert node feature into a properly formatted launcher argument.
        """
        logger.warning(f"set_node_feature() not supported for {self.launcher_str()}.")
        return None

    def set_walltime(self, walltime: str) -> t.Union[StringArgument,None]:
        """ Convert walltime into a properly formatted launcher argument.
        """
        logger.warning(f"set_walltime() not supported for {self.launcher_str()}.")
        return None

    def set_binding(self, binding: str) -> t.Union[StringArgument,None]:
        """Set binding

        This sets ``--bind``

        :param binding: Binding, e.g. `packed:21`
        """
        logger.warning(f"set_binding() not supported for {self.launcher_str()}.")
        return None

    def set_cpu_binding_type(self, bind_type: str) -> t.Union[StringArgument,None]:
        """Specifies the cores to which MPI processes are bound

        This sets ``--bind-to`` for MPI compliant implementations

        :param bind_type: binding type
        """
        logger.warning(f"set_cpu_binding_type() not supported for {self.launcher_str()}.")
        return None

    def set_task_map(self, task_mapping: str) -> t.Union[StringArgument,None]:
        """Set ``mpirun`` task mapping

        this sets ``--map-by <mapping>``

        For examples, see the man page for ``mpirun``

        :param task_mapping: task mapping
        """
        logger.warning(f"set_task_map() not supported for {self.launcher_str()}.")
        return None

    def set_quiet_launch(self, quiet: bool) -> t.Union[t.Dict[str, None], None]:
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        """
        logger.warning(f"set_quiet_launch() not supported for {self.launcher_str()}.")
        return None

    def format_env_vars(self, env_vars: t.Dict[str, t.Optional[str]]) -> t.Union[t.List[str],None]:
        """Build bash compatible environment variable string for Slurm

        :returns: the formatted string of environment variables
        """
        logger.warning(f"format_env_vars() not supported for {self.launcher_str()}.")
        return None

    def format_launcher_args(self, launcher_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.Union[t.List[str],None]:
        """ Build formatted launch arguments
        """
        logger.warning(f"format_launcher_args() not supported for {self.launcher_str()}.")
        return None

    def format_comma_sep_env_vars(self, env_vars: t.Dict[str, t.Optional[str]]) -> t.Union[t.Tuple[str, t.List[str]],None]:
        """Build environment variable string for Slurm

        Slurm takes exports in comma separated lists
        the list starts with all as to not disturb the rest of the environment
        for more information on this, see the slurm documentation for srun

        :returns: the formatted string of environment variables
        """
        logger.warning(f"format_comma_sep_env_vars() not supported for {self.launcher_str()}.")
        return None

    def set_verbose_launch(self, verbose: bool) -> t.Union[t.Dict[str, None], t.Dict[str, int], None]:
        """Set the job to run in verbose mode

        This sets ``--verbose``

        :param verbose: Whether the job should be run verbosely
        """
        logger.warning(f"set_verbose_launch() not supported for {self.launcher_str()}.")
        return None