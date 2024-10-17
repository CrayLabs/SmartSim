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

from smartsim.log import get_logger

from ...batch_command import BatchSchedulerType
from ...common import StringArgument
from ..batch_arguments import BatchArguments

logger = get_logger(__name__)


class SlurmBatchArguments(BatchArguments):
    """A class to represent the arguments required for submitting batch
    jobs using the sbatch command.
    """

    def scheduler_str(self) -> str:
        """Get the string representation of the scheduler

        :returns: The string representation of the scheduler
        """
        return BatchSchedulerType.Slurm.value

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        format = "HH:MM:SS"

        :param walltime: wall time
        """
        pattern = r"^\d{2}:\d{2}:\d{2}$"
        if walltime and re.match(pattern, walltime):
            self.set("time", str(walltime))
        else:
            raise ValueError("Invalid walltime format. Please use 'HH:MM:SS' format.")

    def set_nodes(self, num_nodes: int) -> None:
        """Set the number of nodes for this batch job

        This sets ``--nodes``.

        :param num_nodes: number of nodes
        """
        self.set("nodes", str(num_nodes))

    def set_account(self, account: str) -> None:
        """Set the account for this batch job

        This sets ``--account``.

        :param account: account id
        """
        self.set("account", account)

    def set_partition(self, partition: str) -> None:
        """Set the partition for the batch job

        This sets ``--partition``.

        :param partition: partition name
        """
        self.set("partition", str(partition))

    def set_queue(self, queue: str) -> None:
        """alias for set_partition

        Sets the partition for the slurm batch job

        :param queue: the partition to run the batch job on
        """
        return self.set_partition(queue)

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of cpus to use per task

        This sets ``--cpus-per-task``

        :param num_cpus: number of cpus to use per task
        """
        self.set("cpus-per-task", str(cpus_per_task))

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
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
        self.set("nodelist", ",".join(host_list))

    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: batch arguments for `sbatch`
        """
        opts = []
        # TODO add restricted here
        for opt, value in self._batch_args.items():
            # attach "-" prefix if argument is 1 character otherwise "--"
            short_arg = len(opt) == 1
            prefix = "-" if short_arg else "--"

            if not value:
                opts += [prefix + opt]
            else:
                if short_arg:
                    opts += [prefix + opt, str(value)]
                else:
                    opts += ["=".join((prefix + opt, str(value)))]
        return opts

    def set(self, key: str, value: str | None) -> None:
        """Set an arbitrary scheduler argument

        :param key: The launch argument
        :param value: A string representation of the value for the launch
            argument (if applicable), otherwise `None`
        """
        # Store custom arguments in the launcher_args
        self._batch_args[key] = value
