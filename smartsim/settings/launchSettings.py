# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations
import typing as t
import copy


from smartsim.log import get_logger
from .._core.utils.helpers import fmt_dict
from .common import validate_env_vars, validate_args, StringArgument
from .launchCommand import LauncherType
from .translators.launch.alps import AprunArgTranslator
from .translators.launch.lsf import JsrunArgTranslator
from .translators.launch.mpi import MpiArgTranslator, MpiexecArgTranslator, OrteArgTranslator
from .translators.launch.pals import PalsMpiexecArgTranslator
from .translators.launch.slurm import SlurmArgTranslator      
from .translators.launch.dragon import DragonArgTranslator  
from .translators.launch.local import LocalArgTranslator 
from .translators import LaunchArgTranslator   
from .baseSettings import BaseSettings                                                        

logger = get_logger(__name__)

class LaunchSettings(BaseSettings):
    def __init__(
        self,
        launcher: t.Union[LauncherType, str],
        launcher_args: t.Optional[t.Dict[str, t.Union[str,int,float,None]]] = None,
        env_vars: t.Optional[StringArgument] = None,
    ) -> None:
        try:
            self._launcher = LauncherType(launcher)
        except KeyError:
            raise ValueError(f"Invalid launcher type: {launcher}")
        self._arg_translator = self._get_launcher()
        
        if env_vars:
            validate_env_vars(env_vars)
        self.env_vars = env_vars or {}

        if launcher_args:
            validate_args(launcher_args)
        self.launcher_args = launcher_args or {}
    
    @property
    def launcher(self):
        return self._launcher

    @property
    def arg_translator(self):
        return self._arg_translator

    @property
    def launcher_args(self) -> t.Dict[str, t.Union[int, str, float, None]]:
        """Return an immutable list of attached launcher arguments.

        :returns: attached run arguments
        """
        return self._launcher_args

    @launcher_args.setter
    def launcher_args(self, value: t.Dict[str, t.Union[int, str, float,None]]) -> None:
        """Set the launcher arguments.

        :param value: run arguments
        """
        self._launcher_args = copy.deepcopy(value)

    @property
    def env_vars(self) -> StringArgument:
        """Return an immutable list of attached environment variables.

        :returns: attached environment variables
        """
        return self._env_vars

    @env_vars.setter
    def env_vars(self, value: StringArgument) -> None:
        """Set the environment variables.

        :param value: environment variables
        """
        self._env_vars = copy.deepcopy(value)
    
    @property
    def reserved_launch_args(self):
        return self.arg_translator.set_reserved_launch_args()
    
    def _get_launcher(self) -> LaunchArgTranslator:
        """ Map the Launcher to the LaunchArgTranslator
        """
        if self._launcher.value == 'slurm':
            return SlurmArgTranslator()
        elif self._launcher.value == 'mpiexec':
            return MpiexecArgTranslator()
        elif self._launcher.value == 'mpirun':
            return MpiArgTranslator()
        elif self._launcher.value == 'orterun':
            return OrteArgTranslator()
        elif self._launcher.value == 'alps':
            return AprunArgTranslator()
        elif self._launcher.value == 'lsf':
            return JsrunArgTranslator()
        elif self._launcher.value == 'pals':
            return PalsMpiexecArgTranslator()
        elif self._launcher.value == 'dragon':
            return DragonArgTranslator()
        elif self._launcher.value == 'local':
            return LocalArgTranslator()

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
            self.update_launcher_args(args)

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """ Specify the hostlist for this job

        :param host_list: hosts to launch on
        """

        args = self.arg_translator.set_hostlist(host_list)

        if args:
            self.update_launcher_args(args)

    def set_hostlist_from_file(self, file_path: str) -> None:
        """ Use the contents of a file to set the node list

        :param file_path: Path to the hostlist file
        """
        
        args = self.arg_translator.set_hostlist_from_file(file_path)
        
        if args:
            self.update_launcher_args(args)

    def set_excluded_hosts(self, host_list: t.Union[str, t.List[str]]) -> None:
        """ Specify a list of hosts to exclude for launching this job

        :param host_list: hosts to exclude
        """
        
        args = self.arg_translator.set_excluded_hosts(host_list)

        if args:
            self.update_launcher_args(args)
    
    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """ Set the number of cpus to use per task

        :param cpus_per_task: number of cpus to use per task
        """
        
        args = self.arg_translator.set_cpus_per_task(cpus_per_task)

        if args:
            self.update_launcher_args(args)

    def set_tasks(self, tasks: int) -> None:
        """ Set the number of tasks for this job

        :param tasks: number of tasks
        """
        args = self.arg_translator.set_tasks(tasks)
        if args:
            self.update_launcher_args(args)

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """ Set the number of tasks per node for this job

        :param tasks_per_node: number of tasks per node
        """
        args = self.arg_translator.set_tasks_per_node(tasks_per_node)
        if args:
            self.update_launcher_args(args)

    def set_cpu_bindings(self, bindings: t.Union[int, t.List[int]]) -> None:
        """ Bind by setting CPU masks on tasks

        :param bindings: List specifing the cores to which MPI processes are bound
        """
        args = self.arg_translator.set_cpu_bindings(bindings)
        if args:
            self.update_launcher_args(args)

    def set_memory_per_node(self, memory_per_node: int) -> None:
        """ Specify the real memory required per node

        :param memory_per_node: Amount of memory per node in megabytes
        """
        args = self.arg_translator.set_memory_per_node(memory_per_node)
        if args:
            self.update_launcher_args(args)

    def set_executable_broadcast(self, dest_path: str) -> None:
        """ Copy executable file to allocated compute nodes

        :param dest_path: Path to copy an executable file
        """
        args = self.arg_translator.set_executable_broadcast(dest_path)
        if args:
            self.update_launcher_args(args)

    def set_node_feature(self, feature_list: t.Union[str, t.List[str]]) -> None:
        """Specify the node feature for this job

        :param feature_list: node feature to launch on
        :raises TypeError: if not str or list of str
        """
        args = self.arg_translator.set_node_feature(feature_list)
        if args:
            self.update_launcher_args(args)

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        :param walltime: wall time
        """
        args = self.arg_translator.set_walltime(walltime)
        if args:
            self.update_launcher_args(args)

    def set_binding(self, binding: str) -> None:
        """Set binding

        This sets ``--bind``

        :param binding: Binding, e.g. `packed:21`
        """
        args = self.arg_translator.set_binding(binding)
        if args:
            self.update_launcher_args(args)

    def set_cpu_binding_type(self, bind_type: str) -> None:
        """Specifies the cores to which MPI processes are bound

        This sets ``--bind-to`` for MPI compliant implementations

        :param bind_type: binding type
        """
        args = self.arg_translator.set_cpu_binding_type(bind_type)
        if args:
            self.update_launcher_args(args)

    def set_task_map(self, task_mapping: str) -> None:
        """Set ``mpirun`` task mapping

        this sets ``--map-by <mapping>``

        For examples, see the man page for ``mpirun``

        :param task_mapping: task mapping
        """
        args = self.arg_translator.set_task_map(task_mapping)
        if args:
            self.update_launcher_args(args)

    def set_het_group(self, het_group: t.Iterable[int]) -> None:
        """Set the heterogeneous group for this job

        this sets `--het-group`

        :param het_group: list of heterogeneous groups
        """
        args = self.arg_translator.set_het_group(het_group)
        if args:
            self.update_launcher_args(args)

    def set_verbose_launch(self, verbose: bool) -> None:
        """Set the job to run in verbose mode

        This sets ``--verbose``

        :param verbose: Whether the job should be run verbosely
        """
        args = self.arg_translator.set_verbose_launch(verbose)
        if args and verbose:
            self.update_launcher_args(args)
        if args and not verbose:
            self.launcher_args.pop(next(iter(args)))

    def set_quiet_launch(self, quiet: bool) -> None:
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        """
        args = self.arg_translator.set_quiet_launch(quiet)
        if args and quiet:
            self.update_launcher_args(args)
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

    def update_launcher_args(self, args: t.Mapping[str, int | str | float | None]) -> None:
        self.launcher_args.update(args)
    
    def set(self, key: str, arg: t.Union[str,int,float,None]) -> None:
        # Store custom arguments in the launcher_args
        if not isinstance(key, str):
            raise TypeError("Argument name should be of type str")
        key = key.strip().lstrip("-")
        if key in self.reserved_launch_args:
            logger.warning(
                (
                    f"Could not set argument '{key}': "
                    f"it is a reserved argument of '{type(self).__name__}'"
                )
            )
            return
        if key in self.launcher_args and arg != self.launcher_args[key]:
            logger.warning(f"Overwritting argument '{key}' with value '{arg}'")
        self.launcher_args[key] = arg
    
    def __str__(self) -> str:  # pragma: no-cover
        string = f"\nLauncher: {self.arg_translator.launcher_str}"
        if self.launcher_args:
            string += f"\nLaunch Arguments:\n{fmt_dict(self.launcher_args)}"
        if self.env_vars:
            string += f"\nEnvironment variables: \n{fmt_dict(self.env_vars)}"
        return string