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

from abc import ABC, abstractmethod
import typing as t
from ..common import IntegerArgument, StringArgument

from smartsim.log import get_logger                                                                                    

logger = get_logger(__name__)

class BatchArgTranslator(ABC):
    """Abstract base class that defines all generic scheduler
    argument methods that are not supported.  It is the
    responsibility of child classes for each launcher to translate
    the input parameter to a properly formatted launcher argument.
    """

    @abstractmethod
    def scheduler_str(self) -> str:
        """ Get the string representation of the launcher
        """
        pass

    @abstractmethod
    def set_account(self, account: str) -> t.Union[StringArgument,None]:
        """Set the account for this batch job

        :param account: account id
        """
        pass

    def set_partition(self, partition: str) -> t.Union[StringArgument,None]:
        """Set the partition for the batch job

        :param partition: partition name
        """
        logger.warning(f"set_partition() not supported for {self.scheduler_str()}.")
        return None

    @abstractmethod
    def set_queue(self, queue: str) -> t.Union[StringArgument,None]:
        """alias for set_partition

        Sets the partition for the slurm batch job

        :param queue: the partition to run the batch job on
        """
        pass

    def set_smts(self, smts: int) -> t.Union[IntegerArgument,None]:
        """Set SMTs

        This sets ``-alloc_flags``. If the user sets
        SMT explicitly through ``-alloc_flags``, then that
        takes precedence.

        :param smts: SMT (e.g on Summit: 1, 2, or 4)
        """
        logger.warning(f"set_smts() not supported for {self.scheduler_str()}.")
        return None
    
    def set_project(self, project: str) -> t.Union[StringArgument,None]:
        """Set the project

        This sets ``-P``.

        :param time: project name
        """
        logger.warning(f"set_project() not supported for {self.scheduler_str()}.")
        return None

    @abstractmethod
    def set_walltime(self, walltime: str) -> t.Union[StringArgument,None]:
        """Set the walltime of the job

        :param walltime: wall time
        """
        pass

    @abstractmethod
    def set_nodes(self, num_nodes: int) -> t.Union[IntegerArgument,None]:
        """Set the number of nodes for this batch job

        :param num_nodes: number of nodes
        """
        pass

    def set_cpus_per_task(self, cpus_per_task: int) -> t.Union[IntegerArgument,None]:
        """Set the number of cpus to use per task

        :param num_cpus: number of cpus to use per task
        """
        logger.warning(f"set_cpus_per_task() not supported for {self.scheduler_str()}.")
        return None

    @abstractmethod
    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> t.Union[StringArgument,None]:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :raises TypeError: if not str or list of str
        """
        pass

    @abstractmethod
    def format_batch_args(self, batch_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Sbatch
        """
        pass

    def set_tasks(self, tasks: int) -> t.Union[IntegerArgument,None]:
        """Set the number of tasks for this job

        :param tasks: number of tasks
        """
        logger.warning(f"set_tasks() not supported for {self.scheduler_str()}.")
        return None
    
    def set_ncpus(self, num_cpus: int) -> t.Union[IntegerArgument,None]:
        """Set the number of cpus obtained in each node.

        If a select argument is provided in
        ``QsubBatchSettings.resources``, then
        this value will be overridden

        :param num_cpus: number of cpus per node in select
        """
        logger.warning(f"set_ncpus() not supported for {self.scheduler_str()}.")
        return None