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

from .orchestrator import Orchestrator


class SlurmOrchestrator(Orchestrator):
    def __init__(
        self,
        port=6379,
        db_nodes=1,
        batch=True,
        hosts=None,
        run_command="srun",
        account=None,
        time=None,
        alloc=None,
        db_per_host=1,
        interface="ipogif0",
        single_cmd=False,
        **kwargs,
    ):

        """Initialize an Orchestrator reference for Slurm based systems

        The orchestrator launches as a batch by default. The Slurm orchestrator
        can also be given an allocation to run on. If no allocation is provided,
        and batch=False, at launch, the orchestrator will look for an interactive
        allocation to launch on.

        The SlurmOrchestrator port provided will be incremented if multiple
        databases per node are launched.

        SlurmOrchestrator supports launching with both ``srun`` and ``mpirun``
        as launch binaries. If mpirun is used, the hosts parameter should be
        populated with length equal to that of the ``db_nodes`` argument.

        :param port: TCP/IP port
        :type port: int
        :param db_nodes: number of database shards, defaults to 1
        :type db_nodes: int, optional
        :param batch: Run as a batch workload, defaults to True
        :type batch: bool, optional
        :param hosts: specify hosts to launch on
        :type hosts: list[str]
        :param run_command: specify launch binary. Options are "mpirun" and "srun", defaults to "srun"
        :type run_command: str, optional
        :param account: account to run batch on
        :type account: str, optional
        :param time: walltime for batch 'HH:MM:SS' format
        :type time: str, optional
        :param alloc: allocation to launch on, defaults to None
        :type alloc: str, optional
        :param db_per_host: number of database shards per system host (MPMD), defaults to 1
        :type db_per_host: int, optional
        :param single_cmd: run all shards with one (MPMD) command, defaults to True
        :type single_cmd: bool
        """
        simplefilter("always", DeprecationWarning)
        msg = "SlurmOrchestrator(...) is deprecated and will be removed in a future release.\n"
        msg += "Please update your code to use Orchestrator(launcher='slurm', ...)."
        warn(msg, DeprecationWarning)
        super().__init__(
            port,
            interface,
            db_nodes=db_nodes,
            batch=batch,
            run_command=run_command,
            alloc=alloc,
            db_per_host=db_per_host,
            single_cmd=single_cmd,
            launcher="slurm",
            account=account,
            hosts=hosts,
            time=time,
            **kwargs,
        )
