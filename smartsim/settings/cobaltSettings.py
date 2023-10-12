# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import typing as t
from .base import BatchSettings


class CobaltBatchSettings(BatchSettings):
    def __init__(
        self,
        nodes: t.Optional[int] = None,
        time: str = "",
        queue: t.Optional[str] = None,
        account: t.Optional[str] = None,
        batch_args: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ) -> None:
        """Specify settings for a Cobalt ``qsub`` batch launch

        If the argument doesn't have a parameter, put None
        as the value. e.g. {'exclusive': None}

        Initialization values provided (nodes, time, account)
        will overwrite the same arguments in ``batch_args`` if present

        :param nodes: number of nodes, defaults to None
        :type nodes: int, optional
        :param time: walltime for job, e.g. "10:00:00" for 10 hours,
            defaults to empty str
        :type time: str, optional
        :param queue: queue to launch job in, defaults to None
        :type queue: str, optional
        :param account: account for job, defaults to None
        :type account: str, optional
        :param batch_args: extra batch arguments, defaults to None
        :type batch_args: dict[str, str], optional
        """
        super().__init__(
            "qsub",
            batch_args=batch_args,
            nodes=nodes,
            account=account,
            queue=queue,
            time=time,
            **kwargs,
        )

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        format = "HH:MM:SS"

        Cobalt walltime can also be specified with number
        of minutes.

        :param walltime: wall time
        :type walltime: str
        """
        # TODO check for formatting errors here
        # TODO catch existing "t" in batch_args
        if walltime:
            self.batch_args["time"] = walltime

    def set_nodes(self, num_nodes: int) -> None:
        """Set the number of nodes for this batch job

        :param num_nodes: number of nodes
        :type num_nodes: int
        """
        # TODO catch existing "n" in batch_args
        if num_nodes:
            self.batch_args["nodecount"] = str(int(num_nodes))

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :type host_list: str | list[str]
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all(isinstance(host, str) for host in host_list):
            raise TypeError("host_list argument must be list of strings")
        hosts = ",".join(host_list)
        self.batch_args["attrs"] = f"location={hosts}"

    def set_tasks(self, num_tasks: int) -> None:
        """Set total number of processes to start

        :param num_tasks: number of processes
        :type num_tasks: int
        """
        self.batch_args["proccount"] = str(int(num_tasks))

    def set_queue(self, queue: str) -> None:
        """Set the queue for the batch job

        :param queue: queue name
        :type queue: str
        """
        # TODO catch existing "q" in batch args
        if queue:
            self.batch_args["queue"] = str(queue)

    def set_account(self, account: str) -> None:
        """Set the account for this batch job

        :param acct: account id
        :type acct: str
        """
        # TODO catch existing "A" in batch_args
        if account:
            self.batch_args["project"] = account

    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: list of batch arguments for Sbatch
        :rtype: list[str]
        """
        restricted = [
            "o",
            "output",  # output is determined by interface
            "O",
            "outputprefix",  # step name is output prefix
            "e",
            "error",  # error is determined by interface
            "cwd",  # cwd is determined by interface
            "jobname",  # step name is jobname
        ]
        opts = []
        for opt, value in self.batch_args.items():
            if opt not in restricted:
                # attach "-" prefix if argument is 1 character otherwise "--"
                short_arg = bool(len(str(opt)) == 1)
                prefix = "-" if short_arg else "--"
                if not value:
                    opts += [prefix + opt]
                else:
                    if short_arg:
                        opts += [prefix + opt, str(value)]
                    else:
                        opts += [" ".join((prefix + opt, str(value)))]
        return opts
