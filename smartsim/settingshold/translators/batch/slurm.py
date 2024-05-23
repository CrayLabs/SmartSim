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

import re
import typing as t
from ..batchArgTranslator import BatchArgTranslator
from ...common import IntegerArgument, StringArgument 
from ...batchCommand import SchedulerType
from smartsim.log import get_logger                                                                                

logger = get_logger(__name__)

class SlurmBatchArgTranslator(BatchArgTranslator):

    def scheduler_str(self) -> str:
        """ Get the string representation of the scheduler
        """
        return SchedulerType.SlurmScheduler.value

    def set_walltime(self, walltime: str) -> t.Union[StringArgument,None]:
        """Set the walltime of the job

        format = "HH:MM:SS"

        :param walltime: wall time
        """
        pattern = r'^\d{2}:\d{2}:\d{2}$'
        if walltime and re.match(pattern, walltime):
            return {"time": str(walltime)}
        else:
            raise ValueError("Invalid walltime format. Please use 'HH:MM:SS' format.")

    def set_nodes(self, num_nodes: int) -> t.Union[IntegerArgument,None]:
        """Set the number of nodes for this batch job
        
        This sets ``--nodes``.

        :param num_nodes: number of nodes
        """
        return {"nodes": int(num_nodes)}

    def set_account(self, account: str) -> t.Union[StringArgument,None]:
        """Set the account for this batch job
        
        This sets ``--account``.

        :param account: account id
        """
        return {"account": account}

    def set_partition(self, partition: str) -> t.Union[StringArgument,None]:
        """Set the partition for the batch job
        
        This sets ``--partition``.

        :param partition: partition name
        """
        return {"partition": str(partition)}

    def set_queue(self, queue: str) -> t.Union[StringArgument,None]:
        """alias for set_partition

        Sets the partition for the slurm batch job

        :param queue: the partition to run the batch job on
        """
        return self.set_partition(queue)

    def set_cpus_per_task(self, cpus_per_task: int) -> t.Union[IntegerArgument,None]:
        """Set the number of cpus to use per task

        This sets ``--cpus-per-task``

        :param num_cpus: number of cpus to use per task
        """
        return {"cpus-per-task": int(cpus_per_task)}

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> t.Union[StringArgument,None]:
        """Specify the hostlist for this job
        
        This sets ``--nodelist``.

        :param host_list: hosts to launch on
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        return {"nodelist": ",".join(host_list)}

    def format_batch_args(self, batch_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: batch arguments for Sbatch
        """
        opts = []
        # TODO add restricted here
        for opt, value in batch_args.items():
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