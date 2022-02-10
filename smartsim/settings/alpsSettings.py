# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

from .base import RunSettings
from ..error import SSUnsupportedError

class AprunSettings(RunSettings):
    def __init__(self, exe, exe_args=None, run_args=None, env_vars=None, **kwargs):
        """Settings to run job with ``aprun`` command

        ``AprunSettings`` can be used for both the `pbs` and `cobalt`
        launchers.

        :param exe: executable
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: str | list[str], optional
        :param run_args: arguments for run command, defaults to None
        :type run_args: dict[str, str], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(
            exe,
            exe_args,
            run_command="aprun",
            run_args=run_args,
            env_vars=env_vars,
            **kwargs,
        )
        self.mpmd = []

    def make_mpmd(self, aprun_settings):
        """Make job an MPMD job

        This method combines two ``AprunSettings``
        into a single MPMD command joined with ':'

        :param aprun_settings: ``AprunSettings`` instance
        :type aprun_settings: AprunSettings
        """
        if self.colocated_db_settings:
            raise SSUnsupportedError(
                "Colocated models cannot be run as a mpmd workload"
            )
        self.mpmd.append(aprun_settings)

    def set_cpus_per_task(self, cpus_per_task):
        """Set the number of cpus to use per task

        This sets ``--cpus-per-pe``

        :param cpus_per_task: number of cpus to use per task
        :type cpus_per_task: int
        """
        self.run_args["cpus-per-pe"] = int(cpus_per_task)

    def set_tasks(self, tasks):
        """Set the number of tasks for this job

        This sets ``--pes``

        :param tasks: number of tasks
        :type tasks: int
        """
        self.run_args["pes"] = int(tasks)

    def set_tasks_per_node(self, tasks_per_node):
        """Set the number of tasks for this job

        This sets ``--pes-per-node``

        :param tasks_per_node: number of tasks per node
        :type tasks_per_node: int
        """
        self.run_args["pes-per-node"] = int(tasks_per_node)

    def set_hostlist(self, host_list):
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :type host_list: str | list[str]
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all([isinstance(host, str) for host in host_list]):
            raise TypeError("host_list argument must be list of strings")
        self.run_args["node-list"] = ",".join(host_list)

    def set_excluded_hosts(self, host_list):
        """Specify a list of hosts to exclude for launching this job

        :param host_list: hosts to exclude
        :type host_list: list[str]
        :raises TypeError:
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all([isinstance(host, str) for host in host_list]):
            raise TypeError("host_list argument must be list of strings")
        self.run_args["exclude-node-list"] = ",".join(host_list)

    def format_run_args(self):
        """Return a list of ALPS formatted run arguments

        :return: list of ALPS arguments for these settings
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

    def set_walltime(self, walltime):
        """Set the walltime of the job

        format = "HH:MM:SS"

        :param walltime: wall time
        :type walltime: str
        """
        h_m_s = walltime.split(":")
        self.run_args["t"] = str(
            int(h_m_s[0]) * 3600 + int(h_m_s[1]) * 60 + int(h_m_s[2])
        )
