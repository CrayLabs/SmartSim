from __future__ import annotations

import typing as t
import re
import os
from ..launchArgTranslator import LaunchArgTranslator
from ...common import IntegerArgument, StringArgument
from ...launchCommand import LauncherType
from smartsim.log import get_logger                                                                                

logger = get_logger(__name__)


class SlurmArgTranslator(LaunchArgTranslator):

    def launcher_str(self) -> str:
        """ Get the string representation of the launcher
        """
        return LauncherType.SlurmLauncher.value
    
    def _set_reserved_launch_args(self) -> set[str]:
        """ Return reserved launch arguments.
        """
        return {"chdir", "D"}

    def set_nodes(self, nodes: int) -> t.Union[IntegerArgument, None]:
        """ Set the number of nodes

        Effectively this is setting: ``srun --nodes <num_nodes>``

        :param nodes: nodes to launch on
        :return: launcher argument
        """
        return {"nodes": int(nodes)}
    
    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> t.Union[StringArgument,None]:
        """ Specify the hostlist for this job

        This sets ``--nodelist``

        :param host_list: hosts to launch on
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        elif not isinstance(host_list, list):
            raise TypeError("host_list argument must be a string or list of strings")
        elif not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        return {"nodelist": ",".join(host_list)}

    def set_hostlist_from_file(self, file_path: str) -> t.Union[StringArgument, None]:
        """ Use the contents of a file to set the node list

        This sets ``--nodefile``

        :param file_path: Path to the nodelist file
        """
        return {"nodefile": file_path}

    def set_excluded_hosts(self, host_list: t.Union[str, t.List[str]]) ->  t.Union[StringArgument,None]:
        """ Specify a list of hosts to exclude for launching this job

        :param host_list: hosts to exclude
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        return { "exclude": ",".join(host_list)}

    def set_cpus_per_task(self, cpus_per_task: int) -> t.Union[IntegerArgument,None]:
        """ Set the number of cpus to use per task

        This sets ``--cpus-per-task``

        :param num_cpus: number of cpus to use per task
        """
        return {"cpus-per-task": int(cpus_per_task)}

    def set_tasks(self, tasks: int) -> t.Union[IntegerArgument,None]:
        """ Set the number of tasks for this job

        This sets ``--ntasks``

        :param tasks: number of tasks
        """
        return {"ntasks": int(tasks)}
    
    def set_tasks_per_node(self, tasks_per_node: int) -> t.Union[IntegerArgument,None]:
        """ Set the number of tasks for this job

        This sets ``--ntasks-per-node``

        :param tasks_per_node: number of tasks per node
        """
        return {"ntasks-per-node": int(tasks_per_node)}
    
    def set_cpu_bindings(self, bindings: t.Union[int,t.List[int]]) -> t.Union[StringArgument,None]:
        """ Bind by setting CPU masks on tasks

        This sets ``--cpu-bind`` using the ``map_cpu:<list>`` option

        :param bindings: List specifing the cores to which MPI processes are bound
        """
        if isinstance(bindings, int):
            bindings = [bindings]
        return {"cpu_bind": "map_cpu:" + ",".join(str(num) for num in bindings)}

    def set_memory_per_node(self, memory_per_node: int) -> t.Union[StringArgument,None]:
        """ Specify the real memory required per node

        This sets ``--mem`` in megabytes

        :param memory_per_node: Amount of memory per node in megabytes
        """
        return {"mem": f"{memory_per_node}M"}

    def set_executable_broadcast(self, dest_path: str) -> t.Union[StringArgument,None]:
        """ Copy executable file to allocated compute nodes

        This sets ``--bcast``

        :param dest_path: Path to copy an executable file
        """
        return {"bcast": dest_path}

    def set_node_feature(self, feature_list: t.Union[str, t.List[str]]) -> t.Union[StringArgument,None]:
        """ Specify the node feature for this job

        This sets ``-C``

        :param feature_list: node feature to launch on
        :raises TypeError: if not str or list of str
        """
        if isinstance(feature_list, str):
            feature_list = [feature_list.strip()]
        elif not all(isinstance(feature, str) for feature in feature_list):
            raise TypeError("node_feature argument must be string or list of strings")
        return {"C": ",".join(feature_list)}

    def set_walltime(self, walltime: str) -> t.Union[StringArgument,None]:
        """ Set the walltime of the job

        format = "HH:MM:SS"

        :param walltime: wall time
        """
        pattern = r'^\d{2}:\d{2}:\d{2}$'
        if walltime and re.match(pattern, walltime):
            return {"time": str(walltime)}
        else:
            raise ValueError("Invalid walltime format. Please use 'HH:MM:SS' format.")

    def set_verbose_launch(self, verbose: bool) -> t.Union[t.Dict[str, None], t.Dict[str, int], None]:
        """ Set the job to run in verbose mode

        This sets ``--verbose``

        :param verbose: Whether the job should be run verbosely
        """
        return {"verbose": None}

    def set_quiet_launch(self, quiet: bool) -> t.Union[t.Dict[str, None], None]:
        """Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        """
        return {"quiet": None}

    def format_launcher_args(self, launcher_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.Union[t.List[str],None]:
        """Return a list of slurm formatted launch arguments

        :return: list of slurm arguments for these settings
        """
        # add additional slurm arguments based on key length
        opts = []
        for opt, value in launcher_args.items():
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
    
    def format_env_vars(self, env_vars: t.Dict[str, t.Optional[str]]) -> t.Union[t.List[str],None]:
        """Build bash compatible environment variable string for Slurm

        :returns: the formatted string of environment variables
        """
        self._check_env_vars(env_vars)
        return [f"{k}={v}" for k, v in env_vars.items() if "," not in str(v)]

    def format_comma_sep_env_vars(self, env_vars: t.Dict[str, t.Optional[str]]) -> t.Union[t.Tuple[str, t.List[str]],None]:
        """Build environment variable string for Slurm

        Slurm takes exports in comma separated lists
        the list starts with all as to not disturb the rest of the environment
        for more information on this, see the slurm documentation for srun

        :returns: the formatted string of environment variables
        """
        self._check_env_vars(env_vars)
        exportable_env, compound_env, key_only = [], [], []

        for k, v in env_vars.items():
            kvp = f"{k}={v}"

            if "," in str(v):
                key_only.append(k)
                compound_env.append(kvp)
            else:
                exportable_env.append(kvp)

        # Append keys to exportable KVPs, e.g. `--export x1=v1,KO1,KO2`
        fmt_exported_env = ",".join(v for v in exportable_env + key_only)

        return fmt_exported_env, compound_env

    def _check_env_vars(self, env_vars: t.Dict[str, t.Optional[str]]) -> None:
        """Warn a user trying to set a variable which is set in the environment

        Given Slurm's env var precedence, trying to export a variable which is already
        present in the environment will not work.
        """
        for k, v in env_vars.items():
            if "," not in str(v):
                # If a variable is defined, it will take precedence over --export
                # we warn the user
                preexisting_var = os.environ.get(k, None)
                if preexisting_var is not None and preexisting_var != v:
                    msg = (
                        f"Variable {k} is set to {preexisting_var} in current "
                        "environment. If the job is running in an interactive "
                        f"allocation, the value {v} will not be set. Please "
                        "consider removing the variable from the environment "
                        "and re-run the experiment."
                    )
                    logger.warning(msg)