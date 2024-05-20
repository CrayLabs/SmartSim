from __future__ import annotations

import typing as t
from ..launchArgTranslator import LaunchArgTranslator
from ...common import IntegerArgument, StringArgument
from ...launchCommand import LauncherType
from smartsim.log import get_logger                                                                                

logger = get_logger(__name__)

class PalsMpiexecArgTranslator(LaunchArgTranslator):

    def launcher_str(self) -> str:
        """ Get the string representation of the launcher
        """
        return LauncherType.PalsLauncher.value

    def _set_reserved_launch_args(self) -> set[str]:
        return set()

    def set_cpu_binding_type(self, bind_type: str) -> t.Union[StringArgument,None]:
        """ Specifies the cores to which MPI processes are bound

        This sets ``--bind-to`` for MPI compliant implementations

        :param bind_type: binding type
        """
        return {"bind-to": bind_type}

    def set_tasks(self, tasks: int) -> t.Union[IntegerArgument, None]:
        """ Set the number of tasks

        :param tasks: number of total tasks to launch
        """
        return {"np": int(tasks)}

    def set_executable_broadcast(self, dest_path: str) -> t.Union[StringArgument, None]:
        """Copy the specified executable(s) to remote machines

        This sets ``--transfer``

        :param dest_path: Destination path (Ignored)
        """
        return {"transfer": str(dest_path)}

    def set_tasks_per_node(self, tasks_per_node: int) -> t.Union[IntegerArgument, None]:
        """ Set the number of tasks per node
    
        This sets ``--ppn``

        :param tasks_per_node: number of tasks to launch per node
        """
        return {"ppn": int(tasks_per_node)}

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> t.Union[StringArgument, None]:
        """ Set the hostlist for the PALS ``mpiexec`` command

        This sets ``hosts``

        :param host_list: list of host names
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        return {"hosts": ",".join(host_list)}

    def format_launch_args(self, launcher_args: t.Dict[str, t.Union[str,int,float]]) -> t.Union[t.List[str],None]:
        """ Return a list of MPI-standard formatted run arguments

        :return: list of MPI-standard arguments for these settings
        """
        # args launcher uses
        args = []
        restricted = ["wdir", "wd"]

        for opt, value in launcher_args.items():
            if opt not in restricted:
                prefix = "--"
                if not value:
                    args += [prefix + opt]
                else:
                    args += [prefix + opt, str(value)]

        return args

    def format_env_vars(self, env_vars: t.Optional[t.Dict[str, t.Optional[str]]]) -> t.Union[t.List[str],None]:
        """ Format the environment variables for mpirun

        :return: list of env vars
        """
        formatted = []

        export_vars = []
        if env_vars:
            for name, value in env_vars.items():
                if value:
                    formatted += ["--env", "=".join((name, str(value)))]
                else:
                    export_vars.append(name)

        if export_vars:
            formatted += ["--envlist", ",".join(export_vars)]

        return formatted

    def format_launcher_args(self, launch_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.List[str]:
        """Return a list of MPI-standard formatted launcher arguments

        :return: list of MPI-standard arguments for these settings
        """
        # args launcher uses
        args = []
        restricted = ["wdir", "wd"]

        for opt, value in launch_args.items():
            if opt not in restricted:
                prefix = "--"
                if not value:
                    args += [prefix + opt]
                else:
                    args += [prefix + opt, str(value)]

        return args