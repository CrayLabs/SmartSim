from __future__ import annotations
from enum import Enum
import typing as t
import copy


from smartsim.log import get_logger

from .launchCommand import LauncherType
from .translators.launch.alps import AprunArgTranslator
from .translators.launch.lsf import JsrunArgTranslator
from .translators.launch.mpi import MpiArgTranslator, MpiexecArgTranslator, OrteArgTranslator
from .translators.launch.pals import PalsMpiexecArgTranslator
from .translators.launch.slurm import SlurmArgTranslator      
from .translators.launch.dragon import DragonArgTranslator  
from .translators.launch.local import LocalArgTranslator 
from .translators import LaunchArgTranslator 

from .common import process_env_vars, IntegerArgument, StringArgument, FloatArgument                                                           

logger = get_logger(__name__)

class SupportedLaunchers(Enum):
    """ Launchers that are supported by
    SmartSim.
    """
    local = "local"
    dragon = "dragon"
    slurm = "slurm"
    mpiexec = "mpiexec"
    mpirun = "mpirun"
    orterun = "orterun"
    aprun = "aprun"
    jsrun = "jsrun"
    pals = "pals"

class LaunchSettings():
    def __init__(
        self,
        launcher: LauncherType,
        launcher_args: t.Optional[t.Dict[str, t.Union[str,int,float,None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ) -> None:
        launcher_to_translator : t.Dict[str,LaunchArgTranslator] = {
            'slurm' : SlurmArgTranslator(),
            'mpiexec' : MpiexecArgTranslator(),
            'mpirun' : MpiArgTranslator(),
            'orterun' : OrteArgTranslator(),
            'aprun' : AprunArgTranslator(),
            'jsrun' : JsrunArgTranslator(),
            'pals' : PalsMpiexecArgTranslator(),
            'dragon': DragonArgTranslator(),
            'local': LocalArgTranslator(),
        }
        if launcher in launcher_to_translator:
            self.launcher = launcher
        else:
            raise ValueError(f"'{launcher}' is not a valid launcher name.")

        process_env_vars(env_vars)
        self.env_vars = env_vars or {}

        # TODO check and preporcess launcher_args
        self.launcher_args = launcher_args or {}

        self.arg_translator = t.cast(LaunchArgTranslator,launcher_to_translator.get(launcher))

    @property
    def launcher_args(self) -> t.Dict[str, t.Union[int, str, float, None]]:
        """Return an immutable list of attached run arguments.

        :returns: attached run arguments
        """
        return self._launcher_args

    @launcher_args.setter
    def launcher_args(self, value: t.Dict[str, t.Union[int, str, float,None]]) -> None:
        """Set the run arguments.

        :param value: run arguments
        """
        self._launcher_args = copy.deepcopy(value)

    def launcher_str(self) -> str:
        """ Get the string representation of the launcher
        """
        return self.arg_translator.launcher_str()
    
    def set_nodes(self, nodes: int) -> None:
        """ Sets the number of nodes

        :param nodes: The number of nodes
        """

        args = self.arg_translator.set_nodes(nodes)

        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """ Specify the hostlist for this job

        :param host_list: hosts to launch on
        """

        args = self.arg_translator.set_hostlist(host_list)

        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_hostlist_from_file(self, file_path: str) -> None:
        """ Use the contents of a file to set the node list

        :param file_path: Path to the hostlist file
        """
        
        args = self.arg_translator.set_hostlist_from_file(file_path)
        
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_excluded_hosts(self, host_list: t.Union[str, t.List[str]]) -> None:
        """ Specify a list of hosts to exclude for launching this job

        :param host_list: hosts to exclude
        """
        
        args = self.arg_translator.set_excluded_hosts(host_list)

        if args:
            for key, value in args.items():
                self.set(key, value)
    
    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """ Set the number of cpus to use per task

        :param cpus_per_task: number of cpus to use per task
        """
        
        args = self.arg_translator.set_cpus_per_task(cpus_per_task)

        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_tasks(self, tasks: int) -> None:
        """ Set the number of tasks for this job

        :param tasks: number of tasks
        """
        args = self.arg_translator.set_tasks(tasks)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """ Set the number of tasks per node for this job

        :param tasks_per_node: number of tasks per node
        """
        args = self.arg_translator.set_tasks_per_node(tasks_per_node)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_cpu_bindings(self, bindings: t.Union[int, t.List[int]]) -> None:
        """ Bind by setting CPU masks on tasks

        :param bindings: List specifing the cores to which MPI processes are bound
        """
        args = self.arg_translator.set_cpu_bindings(bindings)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_memory_per_node(self, memory_per_node: int) -> None:
        """ Specify the real memory required per node

        :param memory_per_node: Amount of memory per node in megabytes
        """
        args = self.arg_translator.set_memory_per_node(memory_per_node)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_executable_broadcast(self, dest_path: str) -> None:
        """ Copy executable file to allocated compute nodes

        :param dest_path: Path to copy an executable file
        """
        args = self.arg_translator.set_executable_broadcast(dest_path)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_node_feature(self, feature_list: t.Union[str, t.List[str]]) -> None:
        """Specify the node feature for this job

        :param feature_list: node feature to launch on
        :raises TypeError: if not str or list of str
        """
        args = self.arg_translator.set_node_feature(feature_list)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        :param walltime: wall time
        """
        args = self.arg_translator.set_walltime(walltime)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_binding(self, binding: str) -> None:
        """Set binding

        This sets ``--bind``

        :param binding: Binding, e.g. `packed:21`
        """
        args = self.arg_translator.set_binding(binding)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_cpu_binding_type(self, bind_type: str) -> None:
        """Specifies the cores to which MPI processes are bound

        This sets ``--bind-to`` for MPI compliant implementations

        :param bind_type: binding type
        """
        args = self.arg_translator.set_cpu_binding_type(bind_type)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_task_map(self, task_mapping: str) -> None:
        """Set ``mpirun`` task mapping

        this sets ``--map-by <mapping>``

        For examples, see the man page for ``mpirun``

        :param task_mapping: task mapping
        """
        args = self.arg_translator.set_task_map(task_mapping)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_verbose_launch(self, verbose: bool) -> None:
        """Set the job to run in verbose mode

        This sets ``--verbose``

        :param verbose: Whether the job should be run verbosely
        """
        args = self.arg_translator.set_verbose_launch(verbose)
        if args and verbose:
            for key, value in args.items():
                self.set(key, value)
        if args and not verbose:
            self.launcher_args.pop(next(iter(args)))

    def set_quiet_launch(self, quiet: bool) -> None:
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        """
        args = self.arg_translator.set_quiet_launch(quiet)
        if args and quiet:
            for key, value in args.items():
                self.set(key, value)
        if args and not quiet:
            self.launcher_args.pop(next(iter(args)))

    def format_comma_sep_env_vars(self) -> t.Union[t.Tuple[str, t.List[str]],None]:
        """Build environment variable string for Slurm

        Slurm takes exports in comma separated lists
        the list starts with all as to not disturb the rest of the environment
        for more information on this, see the slurm documentation for srun

        :returns: the formatted string of environment variables
        """
        return self.arg_translator.format_comma_sep_env_vars(self.env_vars)

    def format_launcher_args(self) -> t.Union[t.List[str],None]:
        """Return formatted launch arguments

        For ``RunSettings``, the run arguments are passed
        literally with no formatting.

        :return: list run arguments for these settings
        """
        return self.arg_translator.format_launcher_args(self.launcher_args)

    def format_env_vars(self) -> t.Union[t.List[str],None]:
        """Build bash compatible environment variable string for Slurm

        :returns: the formatted string of environment variables
        """
        return self.arg_translator.format_env_vars(self.env_vars)

    def update_env(self, env_vars: t.Dict[str, t.Union[str, int, float, bool]]) -> None:
        """Update the job environment variables

        To fully inherit the current user environment, add the
        workload-manager-specific flag to the launch command through the
        :meth:`add_exe_args` method. For example, ``--export=ALL`` for
        slurm, or ``-V`` for PBS/aprun.


        :param env_vars: environment variables to update or add
        :raises TypeError: if env_vars values cannot be coerced to strings
        """
        val_types = (str, int, float, bool)
        # Coerce env_vars values to str as a convenience to user
        for env, val in env_vars.items():
            if not isinstance(val, val_types):
                raise TypeError(
                    f"env_vars[{env}] was of type {type(val)}, not {val_types}"
                )

            self.env_vars[env] = str(val)

    def set(self, key: str, arg: t.Union[str,int,float,None]) -> None:
        # Store custom arguments in the launcher_args
        if not isinstance(key, str):
            raise TypeError("Argument name should be of type str")
        # if value is not None and not isinstance(value, str):
        #     raise TypeError("Argument value should be of type str or None")
        # arg = arg.strip().lstrip("-")
        # if not condition:
        #     logger.info(f"Could not set argument '{arg}': condition not met")
        #     return
        if key in self.launcher_args and arg != self.launcher_args[key]:
            logger.warning(f"Overwritting argument '{key}' with value '{arg}'")
        self.launcher_args[key] = arg