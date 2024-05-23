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
from ..launchArgTranslator import LaunchArgTranslator
from ...common import IntegerArgument, StringArgument
from ...launchCommand import LauncherType
from smartsim.log import get_logger                                                                                

logger = get_logger(__name__)

class _BaseMPIArgTranslator(LaunchArgTranslator):

    def _set_reserved_launch_args(self) -> set[str]:
        """ Return reserved launch arguments.
        """
        return {"wd", "wdir"}

    def set_task_map(self, task_mapping: str) -> t.Union[StringArgument, None]:
        """ Set ``mpirun`` task mapping

        this sets ``--map-by <mapping>``

        For examples, see the man page for ``mpirun``

        :param task_mapping: task mapping
        """
        return {"map-by": task_mapping}

    def set_cpus_per_task(self, cpus_per_task: int) ->  t.Union[IntegerArgument, None]:
        """ Set the number of tasks for this job

        This sets ``--cpus-per-proc`` for MPI compliant implementations

        note: this option has been deprecated in openMPI 4.0+
        and will soon be replaced.

        :param cpus_per_task: number of tasks
        """
        return {"cpus-per-proc": int(cpus_per_task)}

    def set_executable_broadcast(self, dest_path: str) -> t.Union[StringArgument, None]:
        """Copy the specified executable(s) to remote machines

        This sets ``--preload-binary``

        :param dest_path: Destination path (Ignored)
        """
        if dest_path is not None and isinstance(dest_path, str):
            logger.warning(
                (
                    f"{type(self)} cannot set a destination path during broadcast. "
                    "Using session directory instead"
                )
            )
        return {"preload-binary": dest_path}

    def set_cpu_binding_type(self, bind_type: str) ->  t.Union[StringArgument, None]:
        """ Specifies the cores to which MPI processes are bound

        This sets ``--bind-to`` for MPI compliant implementations

        :param bind_type: binding type
        """
        return {"bind-to": bind_type}

    def set_tasks_per_node(self, tasks_per_node: int) ->  t.Union[IntegerArgument, None]:
        """ Set the number of tasks per node

        :param tasks_per_node: number of tasks to launch per node
        """
        return {"npernode": int(tasks_per_node)}

    def set_tasks(self, tasks: int) ->  t.Union[IntegerArgument, None]:
        """ Set the number of tasks for this job

        This sets ``-n`` for MPI compliant implementations

        :param tasks: number of tasks
        """
        return {"n": int(tasks)}

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) ->  t.Union[StringArgument, None]:
        """ Set the hostlist for the ``mpirun`` command

        This sets ``--host``

        :param host_list: list of host names
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        return {"host": ",".join(host_list)}

    def set_hostlist_from_file(self, file_path: str) ->  t.Union[StringArgument, None]:
        """ Use the contents of a file to set the hostlist

        This sets ``--hostfile``

        :param file_path: Path to the hostlist file
        """
        return {"hostfile": file_path}

    def set_verbose_launch(self, verbose: bool) -> t.Union[t.Dict[str, None], t.Dict[str, int], None]:
        """ Set the job to run in verbose mode

        This sets ``--verbose``

        :param verbose: Whether the job should be run verbosely
        """
        return {"verbose": None}

    def set_walltime(self, walltime: str) -> t.Union[StringArgument, None]:
        """Set the maximum number of seconds that a job will run

        This sets ``--timeout``

        :param walltime: number like string of seconds that a job will run in secs
        """
        return {"timeout": walltime}

    def set_quiet_launch(self, quiet: bool) ->  t.Union[t.Dict[str,None], None]:
        """ Set the job to run in quiet mode

        This sets ``--quiet``

        :param quiet: Whether the job should be run quietly
        """
        return {"quiet": None}

    def format_env_vars(self, env_vars: t.Optional[t.Dict[str, t.Optional[str]]]) -> t.Union[t.List[str],None]:
        """ Format the environment variables for mpirun

        :return: list of env vars
        """
        formatted = []
        env_string = "-x"

        if env_vars:
            for name, value in env_vars.items():
                if value:
                    formatted += [env_string, "=".join((name, str(value)))]
                else:
                    formatted += [env_string, name]
        return formatted
    
    def format_launcher_args(self, launcher_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.List[str]:
        """Return a list of MPI-standard formatted run arguments

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

class MpiArgTranslator(_BaseMPIArgTranslator):

    def launcher_str(self) -> str:
        """ Get the string representation of the launcher
        """
        return LauncherType.MpirunLauncher.value

class MpiexecArgTranslator(_BaseMPIArgTranslator):

    def launcher_str(self) -> str:
        """ Get the string representation of the launcher
        """
        return LauncherType.MpiexecLauncher.value

class OrteArgTranslator(_BaseMPIArgTranslator):

    def launcher_str(self) -> str:
        """ Get the string representation of the launcher
        """
        return LauncherType.OrterunLauncher.value