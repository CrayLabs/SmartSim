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

from ..error import SSUnsupportedError
from .orchestrator import Orchestrator


class LSFOrchestrator(Orchestrator):
    def __init__(
        self,
        port=6379,
        db_nodes=1,
        cpus_per_shard=4,
        gpus_per_shard=0,
        batch=True,
        hosts=None,
        project=None,
        time=None,
        interface="ib0",
        single_cmd=True,
        **kwargs,
    ):

        """Initialize an Orchestrator reference for LSF based systems

        The orchestrator launches as a batch by default. If
        batch=False, at launch, the orchestrator will look for an interactive
        allocation to launch on.

        The LSFOrchestrator port provided will be incremented if multiple
        databases per host are launched (``db_per_host>1``).

        Each database shard is assigned a resource set with cpus and gpus
        allocated contiguously on the host:
        it is the user's responsibility to check if
        enough resources are available on each host.

        A list of hosts to launch the database on can be specified
        these addresses must correspond to
        those of the first ``db_nodes//db_per_host`` compute nodes
        in the allocation: for example, for 8 ``db_nodes`` and 2 ``db_per_host``
        the ``host_list`` must contain the addresses of hosts 1, 2, 3, and 4.

        ``LSFOrchestrator`` is launched with only one ``jsrun`` command
        as launch binary, and an Explicit Resource File (ERF) which is
        automatically generated. The orchestrator is always launched on the
        first ``db_nodes//db_per_host`` compute nodes in the allocation.

        :param port: TCP/IP port
        :type port: int
        :param db_nodes: number of database shards, defaults to 1
        :type db_nodes: int, optional
        :param cpus_per_shard: cpus to allocate per shard, defaults to 4
        :type cpus_per_shard: int, optional
        :param gpus_per_shard: gpus to allocate per shard, defaults to 0
        :type gpus_per_shard: int, optional
        :param batch: Run as a batch workload, defaults to True
        :type batch: bool, optional
        :param hosts: specify hosts to launch on
        :type hosts: list[str], optional
        :param project: project to run batch on
        :type project: str, optional
        :param time: walltime for batch 'HH:MM' format
        :type time: str, optional
        :param interface: network interface to use
        :type interface: str
        """
        simplefilter("always", DeprecationWarning)
        msg = "LSFOrchestrator(...) is deprecated and will be removed in a future release.\n"
        msg += "Please update your code to use Orchestrator(launcher='lsf', ...)."
        warn(msg, DeprecationWarning)
        if single_cmd != True:
            raise SSUnsupportedError(
                "LSFOrchestrator can only be run with single_cmd=True (MPMD)."
            )

        super().__init__(
            port,
            interface,
            db_nodes=db_nodes,
            batch=batch,
            run_command="jsrun",
            launcher="lsf",
            project=project,
            hosts=hosts,
            time=time,
            cpus_per_shard=cpus_per_shard,
            gpus_per_shard=gpus_per_shard,
            **kwargs,
        )
