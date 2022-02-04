# BSD 2-Clause License
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

from warnings import simplefilter, warn

from ..log import get_logger
from .orchestrator import Orchestrator

logger = get_logger(__name__)


class PBSOrchestrator(Orchestrator):
    def __init__(
        self,
        port=6379,
        db_nodes=1,
        batch=True,
        hosts=None,
        run_command="aprun",
        interface="ipogif0",
        account=None,
        time=None,
        queue=None,
        single_cmd=True,
        **kwargs,
    ):
        """Initialize an Orchestrator reference for PBSPro based systems

        The ``PBSOrchestrator`` launches as a batch by default. If batch=False,
        at launch, the ``PBSOrchestrator`` will look for an interactive
        allocation to launch on.

        The PBS orchestrator does not support multiple databases per node.

        If ``mpirun`` is specifed as the ``run_command``, then the ``hosts``
        argument is required.

        :param port: TCP/IP port
        :type port: int
        :param db_nodes: number of compute nodes to span accross, defaults to 1
        :type db_nodes: int, optional
        :param batch: run as a batch workload, defaults to True
        :type batch: bool, optional
        :param hosts: specify hosts to launch on, defaults to None
        :type hosts: list[str]
        :param run_command: specify launch binary. Options are ``mpirun`` and ``aprun``, defaults to "aprun"
        :type run_command: str, optional
        :param interface: network interface to use, defaults to "ipogif0"
        :type interface: str, optional
        :param account: account to run batch on
        :type account: str, optional
        :param time: walltime for batch 'HH:MM:SS' format
        :type time: str, optional
        :param queue: queue to launch batch in
        :type queue: str, optional
        """
        simplefilter("always", DeprecationWarning)
        msg = "PBSOrchestrator(...) is deprecated and will be removed in a future release.\n"
        msg += "Please update your code to use Orchestrator(launcher='pbs', ...)."
        warn(msg, DeprecationWarning)
        super().__init__(
            port,
            interface,
            db_nodes=db_nodes,
            batch=batch,
            run_command=run_command,
            single_cmd=single_cmd,
            launcher="pbs",
            hosts=hosts,
            account=account,
            queue=queue,
            time=time,
            **kwargs,
        )
