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
from .common import process_env_vars, process_args, StringArgument
from .batchCommand import SchedulerType
from .translators.batch.pbs import QsubBatchArgTranslator
from .translators.batch.slurm import SlurmBatchArgTranslator
from .translators.batch.lsf import BsubBatchArgTranslator
from .translators import BatchArgTranslator
from .baseSettings import BaseSettings

logger = get_logger(__name__)

class BatchSettings(BaseSettings):
    def __init__(
        self,
        scheduler: SchedulerType,
        scheduler_args: t.Optional[t.Dict[str, t.Union[str,int,float,None]]] = None,
        env_vars: t.Optional[StringArgument] = None,
        **kwargs: t.Any,
    ) -> None:
        scheduler_to_translator = {
            'sbatch' : SlurmBatchArgTranslator(),
            'bsub' : BsubBatchArgTranslator(),
            'qsub' : QsubBatchArgTranslator(),
        }
        if scheduler.value in scheduler_to_translator:
            self.scheduler = scheduler
        else:
            raise ValueError(f"'{scheduler}' is not a valid scheduler name.")

        if env_vars:
            process_env_vars(env_vars)
        self.env_vars = env_vars or {}

        if scheduler_args:
            process_args(scheduler_args)
        self.scheduler_args = scheduler_args or {}

        self.arg_translator = t.cast(BatchArgTranslator,scheduler_to_translator.get(scheduler.value))

    @property
    def scheduler_args(self) -> t.Dict[str, t.Union[int, str, float, None]]:
        """Retrieve attached batch arguments

        :returns: attached batch arguments
        """
        return self._scheduler_args

    @scheduler_args.setter
    def scheduler_args(self, value: t.Dict[str, t.Union[int, str, float,None]]) -> None:
        """Attach batch arguments

        :param value: dictionary of batch arguments
        """
        self._scheduler_args = copy.deepcopy(value) if value else {}

    def scheduler_str(self) -> str:
        """ Get the string representation of the scheduler
        """
        return self.arg_translator.scheduler_str()

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        format = "HH:MM:SS"

        :param walltime: wall time
        """
        # TODO check for formatting here
        args = self.arg_translator.set_walltime(walltime)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_nodes(self, num_nodes: int) -> None:
        """Set the number of nodes for this batch job

        :param num_nodes: number of nodes
        """
        args = self.arg_translator.set_nodes(num_nodes)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_account(self, account: str) -> None:
        """Set the account for this batch job

        :param account: account id
        """
        args = self.arg_translator.set_account(account)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_partition(self, partition: str) -> None:
        """Set the partition for the batch job

        :param partition: partition name
        """
        args = self.arg_translator.set_partition(partition)
        if args:
            for key, value in args.items():
                self.set(key, value)
    
    def set_queue(self, queue: str) -> None:
        """alias for set_partition

        Sets the partition for the slurm batch job

        :param queue: the partition to run the batch job on
        """
        args = self.arg_translator.set_queue(queue)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of cpus to use per task

        This sets ``--cpus-per-task``

        :param num_cpus: number of cpus to use per task
        """
        args = self.arg_translator.set_cpus_per_task(cpus_per_task)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :raises TypeError: if not str or list of str
        """
        args = self.arg_translator.set_hostlist(host_list)
        if args:
            for key, value in args.items():
                self.set(key, value)
    
    def set_smts(self, smts: int) -> None:
        """Set SMTs

        This sets ``-alloc_flags``. If the user sets
        SMT explicitly through ``-alloc_flags``, then that
        takes precedence.

        :param smts: SMT (e.g on Summit: 1, 2, or 4)
        """
        args = self.arg_translator.set_smts(smts)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_project(self, project: str) -> None:
        """Set the project

        This sets ``-P``.

        :param time: project name
        """
        args = self.arg_translator.set_project(project)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        This sets ``-n``

        :param tasks: number of tasks
        """
        args = self.arg_translator.set_tasks(tasks)
        if args:
            for key, value in args.items():
                self.set(key, value)
    
    def set_ncpus(self, num_cpus: int) -> None:
        """Set the number of cpus obtained in each node.

        If a select argument is provided in
        ``QsubBatchSettings.resources``, then
        this value will be overridden

        :param num_cpus: number of cpus per node in select
        """
        args = self.arg_translator.set_ncpus(num_cpus)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview
        """
        return self.arg_translator.format_batch_args(self.scheduler_args)

    def set(self, key: str, arg: t.Union[str,int,float,None]) -> None:
        # Store custom arguments in the launcher_args
        self.scheduler_args[key] = arg

    def __str__(self) -> str:  # pragma: no-cover
        string = ""
        string += f"\Scheduler: {self.arg_translator.scheduler_str}"
        if self.scheduler_args:
            string += f"\Scheduler Arguments:\n{fmt_dict(self.scheduler_args)}"
        if self.env_vars:
            string += f"\nEnvironment variables: \n{fmt_dict(self.env_vars)}"
        return string