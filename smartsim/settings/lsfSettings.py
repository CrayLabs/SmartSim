git # BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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

from ..error import SSConfigError
from ..utils.helpers import init_default
from .settings import BatchSettings, RunSettings


class JsrunSettings(RunSettings):
    def __init__(self, exe, exe_args=None, run_args=None, env_vars=None):
        """Settings to run job with ``jsrun`` command

        ``JsrunSettings`` can be used for both the `lsf` launcher.

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
            exe, exe_args, run_command="jsrun", run_args=run_args, env_vars=env_vars
        )

    def set_num_rs(self, num_rs):
        """Set the number of resource sets to use

        This sets ``--nrs``. 

        :param num_rs: Number of resource sets or `ALL_HOSTS`
        :type num_rs: int or str
        """

        if isinstance(num_rs, str):
            self.run_args["nrs"] = num_rs
        else:
            self.run_args["nrs"] = int(num_rs)

    def set_cpus_per_rs(self, num_cpus):
        """Set the number of cpus to use per resource set

        This sets ``--cpu_per_rs``

        :param num_cpus: number of cpus to use per resource set
        :type num_cpus: int
        """
        self.run_args["cpu_per_rs"] = int(num_cpus)

    def set_gpus_per_rs(self, num_gpus):
        """Set the number of gpus to use per resource set

        This sets ``--gpu_per_rs``

        :param num_cpus: number of gpus to use per resource set
        :type num_gpus: int
        """
        self.run_args["gpu_per_rs"] = int(num_gpus)

    def set_tasks(self, num_tasks):
        """Set the number of tasks for this job

        This sets ``--np``

        :param num_tasks: number of tasks
        :type num_tasks: int
        """
        self.run_args["np"] = int(num_tasks)

    def set_tasks_per_rs(self, num_tprs):
        """Set the number of tasks per resource set

        This sets ``--tasks_per_rs``

        :param num_tpn: number of tasks per resource set
        :type num_tpn: int
        """
        self.run_args["tasks_per_rs"] = int(num_tprs)

    def format_run_args(self):
        """Return a list of LSF formatted run arguments

        :return: list LSF arguments for these settings
        :rtype: list[str]
        """
        # args launcher uses
        args = []
        restricted = ["chdir"]

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

class BsubBatchSettings(BatchSettings):
    def __init__(
        self,
        nrs=None,
        cpus_per_rs=None,
        gpus_per_rs=None,
        time=None,
        project=None,
        batch_args=None,
        **kwargs,
    ):
        """Specify ``bsub`` batch parameters for a job

        :param nodes: number of nodes for batch
        :type nodes: int, optional
        :param ncpus: number of cpus per node
        :type ncpus: int, optional
        :param time: walltime for batch job in format hh:mm
        :type time: str, optional
        :param queue: queue to run batch in
        :type queue: str
        :param project: project for batch launch
        :type project: str, optional
        :param resources: overrides for resource arguments
        :type resources: dict[str, str], optional
        :param batch_args: overrides for LSF batch arguments
        :type batch_args: dict[str, str], optional
        """
        super().__init__("bsub", batch_args=batch_args)

        if project:
            self.set_project(project)
        if nrs:
            self.set_num_rs(nrs)
        if cpus_per_rs:
            self.set_cpus_per_rs(cpus_per_rs)
        if gpus_per_rs:
            self.set_gpus_per_rs(gpus_per_rs)
        if time:
            self.set_walltime(time)

    def set_walltime(self, time):
        """Set the walltime

        This sets ``-W``.

        :param time: Time in hh:mm format
        :type time: str
        """

        self.run_args["W"] = time

    def set_num_rs(self, num_rs):
        """Set the number of resource sets to use

        This sets ``--nrs``. 

        :param num_rs: Number of resource sets or `ALL_HOSTS`
        :type num_rs: int or str
        """

        if isinstance(num_rs, str):
            self.run_args["nrs"] = num_rs
        else:
            self.run_args["nrs"] = int(num_rs)

    def set_cpus_per_rs(self, num_cpus):
        """Set the number of cpus to use per resource set

        This sets ``--cpu_per_rs``

        :param num_cpus: number of cpus to use per resource set
        :type num_cpus: int
        """
        self.run_args["cpu_per_rs"] = int(num_cpus)

    def set_gpus_per_rs(self, num_gpus):
        """Set the number of gpus to use per resource set

        This sets ``--gpu_per_rs``

        :param num_cpus: number of gpus to use per resource set
        :type num_gpus: int
        """
        self.run_args["gpu_per_rs"] = int(num_gpus)

    def set_tasks(self, num_tasks):
        """Set the number of tasks for this job

        This sets ``--np``

        :param num_tasks: number of tasks
        :type num_tasks: int
        """
        self.run_args["np"] = int(num_tasks)

    def set_tasks_per_rs(self, num_tprs):
        """Set the number of tasks per resource set

        This sets ``--tasks_per_rs``

        :param num_tpn: number of tasks per resource set
        :type num_tpn: int
        """
        self.run_args["tasks_per_rs"] = int(num_tprs)

    def format_batch_args(self):
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Qsub
        :rtype: list[str]
        """
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
