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

from shlex import split as sh_split

from ..settings.settings import create_batch_settings, create_run_settings
from ..entity import DBNode
from ..error import SmartSimError, SSUnsupportedError
from ..log import get_logger
from ..settings import SrunSettings, MpirunSettings, AprunSettings
from ..settings import CobaltBatchSettings, SbatchSettings, QsubBatchSettings
from ..wlm import detect_launcher
from .orchestrator import Orchestrator
from .._core.utils.helpers import is_valid_cmd

logger = get_logger(__name__)


class WLMOrchestrator(Orchestrator):
    def __init__(
        self,
        port=6379,
        db_nodes=1,
        batch=True,
        hosts=None,
        launcher="auto",
        run_command="auto",
        account=None,
        time=None,
        alloc=None,
        interface="ipogif0",
        single_cmd=True,
        **kwargs,
    ):

        """Initialize an Orchestrator reference launched through a WLM

        The orchestrator launches as a batch by default. If the WLM
        is Slurn, the orchestrator can also be given an allocation to run on. 
        If no allocation is provided, and batch=False, at launch, the orchestrator 
        will look for an interactive allocation to launch on.

        The WLMOrchestrator port provided will be incremented if multiple
        databases per node are launched.

        WLMOrchestrator supports several launch binaries, if they
        are compatible with the selected (or detected) launcher.
        
        If mpirun is used as run command, the hosts parameter should be
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
        """

        if launcher == "auto":
            launcher = detect_launcher()

        by_launcher = {
                        "slurm": ["srun", "mpirun"],
                        "pbs": ["aprun", "mpirun"],
                        "cobalt": ["aprun", "mpirun"],
                        "lsf": ["jsrun"],
                        }

        def _detect_command(launcher):
            if launcher in by_launcher:
                for cmd in by_launcher[launcher]:
                    if is_valid_cmd(cmd):
                        return cmd
            msg = f"Could not automatically detect a run command to use for launcher {launcher}"
            msg += f"\nSearched for and could not find the following commands: {by_launcher[launcher]}"
            raise SmartSimError(msg)

        if run_command == "auto":
            run_command = _detect_command(launcher)
        else:
            if run_command not in by_launcher[launcher]:
                msg = f"Run command {run_command} is not supported on launcher {launcher}\n"
                msg+= f"Supported run commands for Orchestrator are: {by_launcher[launcher]}"
                raise SmartSimError(msg)

        self.launcher = launcher
        self.run_command = run_command
        super().__init__(
            port=port,
            interface=interface,
            db_nodes=db_nodes,
            batch=batch,
            launcher=launcher,
            run_command=run_command,
            alloc=alloc,
            single_cmd=single_cmd,
            **kwargs,
        )
        self.batch_settings = self._build_batch_settings(
            db_nodes, alloc, batch, account, time, launcher=launcher, **kwargs
        )
        if hosts:
            self.set_hosts(hosts)
        elif not hosts and run_command == "mpirun":
            raise SmartSimError(
                "hosts argument is required when launching Orchestrator with mpirun"
            )
        self._reserved_run_args = {}
        self._reserved_batch_args = {}
        self._fill_reserved()


    def set_walltime(self, walltime):
        """Set the batch walltime of the orchestrator

        Note: This will only effect orchestrators launched as a batch

        :param walltime: amount of time e.g. 10 hours is 10:00:00
        :type walltime: str
        :raises SmartSimError: if orchestrator isn't launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set walltime")
        self.batch_settings.set_walltime(walltime)

    def set_hosts(self, host_list):
        """Specify the hosts for the ``WLMOrchestrator`` to launch on

        :param host_list: list of host (compute node names)
        :type host_list: str, list[str]
        :raises TypeError: if wrong type
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all([isinstance(host, str) for host in host_list]):
            raise TypeError("host_list argument must be list of strings")
        # TODO check length
        if self.batch:
            self.batch_settings.set_hostlist(host_list)
        
        for host, db in zip(host_list, self.entities):
            if isinstance(db.run_settings, AprunSettings):
                if not self.batch:
                    db.run_settings.set_hostlist([host])
            else:
                db.run_settings.set_hostlist([host])
            if db._mpmd:
                for i, mpmd_runsettings in enumerate(db.run_settings.mpmd):
                    mpmd_runsettings.set_hostlist(host_list[i+1])

    def set_batch_arg(self, arg, value):
        """Set a batch argument the orchestrator should launch with

        Some commonly used arguments such as --job-name are used
        by SmartSim and will not be allowed to be set.

        :param arg: batch argument to set e.g. "exclusive"
        :type arg: str
        :param value: batch param - set to None if no param value
        :type value: str | None
        :raises SmartSimError: if orchestrator not launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set batch_arg")
        if arg in self._reserved_batch_args[type(self.batch_settings)]:
            logger.warning(
                f"Can not set batch argument {arg}: it is a reserved keyword in Orchestrator"
            )
        else:
            self.batch_settings.batch_args[arg] = value

    def set_run_arg(self, arg, value):
        """Set a run argument the orchestrator should launch
        each node with (it will be passed to `jrun`)

        Some commonly used arguments are used
        by SmartSim and will not be allowed to be set.
        For example, "n", "N", etc.

        :param arg: run argument to set
        :type arg: str
        :param value: run parameter - set to None if no parameter value
        :type value: str | None
        """
        if arg in self._reserved_run_args[type(self.entities[0].run_settings)]:
            logger.warning(
                f"Can not set run argument {arg}: it is a reserved keyword in Orchestrator"
            )
        else:
            for db in self.entities:
                db.run_settings.run_args[arg] = value
                if db.mpmd:
                    for mpmd in db.run_settings.mpmd:
                        mpmd.run_args[arg] = value

    def _build_batch_settings(self, db_nodes, alloc, batch, account, time, **kwargs):
        batch_settings = None
        launcher = kwargs.pop("launcher")
        
        # enter this conditional if user has not specified an allocation to run
        # on or if user specified batch=False (alloc will be found through env)
        if not alloc and batch:
            batch_settings = create_batch_settings(launcher,
                nodes=db_nodes, time=time, account=account, **kwargs
            )
            
        return batch_settings

    def _build_run_settings(self, exe, exe_args, **kwargs):
        run_command = kwargs.get("run_command")
        run_args = kwargs.get("run_args", {})
        db_nodes = kwargs.get("db_nodes", 1)
        single_cmd = kwargs.get("single_cmd", True)
        mpmd_nodes = single_cmd and db_nodes>1
    
        if mpmd_nodes:
            run_settings = create_run_settings(exe=exe,
                                               exe_args=exe_args[0],
                                               run_args=run_args.copy(),
                                               **kwargs)
            
            # srun has a different way of running MPMD jobs
            if run_command == "srun":
                run_settings.set_tasks(db_nodes)
            else:
                run_settings.set_tasks(1)
            
            for exe_arg in exe_args[1:]:
                mpmd_run_settings = create_run_settings(exe=exe, 
                                                        exe_args=exe_arg,
                                                        run_args=run_args.copy(),
                                                        **kwargs)
                run_settings.make_mpmd(mpmd_run_settings)
        else:
            run_settings = create_run_settings(exe=exe,
                                               exe_args=exe_args,
                                               run_args=run_args,
                                               **kwargs)
            run_settings.set_tasks(1)
        run_settings.set_tasks_per_node(1)

        if isinstance(run_settings, SrunSettings):
            run_args["nodes"] = 1 if not mpmd_nodes else db_nodes
            if mpmd_nodes:
                run_settings.set_output_suffix("%_t")

        return run_settings

    def _initialize_entities(self, **kwargs):
        """Initialize DBNode instances for the orchestrator."""
        db_nodes = kwargs.get("db_nodes", 1)
        self.db_nodes = db_nodes
        single_cmd = kwargs.get("single_cmd", True)

        mpmd_nodes = single_cmd and db_nodes>1
        
        cluster = not bool(db_nodes < 3)
        if int(db_nodes) == 2:
            raise SSUnsupportedError("Orchestrator does not support clusters of size 2")
        port = kwargs.get("port", 6379)

        if mpmd_nodes:
            exe_args_mpmd = []
        for db_id in range(db_nodes):
            if not mpmd_nodes:
                db_node_name = "_".join((self.name, str(db_id)))
                db_shard_name = db_node_name
            else:
                db_node_name = self.name
                db_shard_name = "_".join((self.name, str(db_id)))

            # create the exe_args list for launching multiple databases
            # per node. also collect port range for dbnode
            start_script_args = [
                self._redis_launch_script,  # redis_starter.py
                f"+ifname={self._interface}",  # pass interface to start script
                "+command",  # command flag for argparser
                self._redis_exe,  # redis-server
                self._redis_conf,  # redis6.conf file
                self._rai_module,  # redisai.so
                "--port",  # redis port
                str(port),  # port number
            ]
            if cluster:
                start_script_args += self._get_cluster_args(db_shard_name, port)

            exe_args = " ".join(start_script_args)

            if not mpmd_nodes:
                # if only launching 1 db_per_host, we don't need a list of exe args lists
                run_settings = self._build_run_settings("python", exe_args, **kwargs)

                node = DBNode(db_node_name, self.path, run_settings, [port])
                self.entities.append(node)
            else:
                exe_args_mpmd.append(sh_split(exe_args))
            
        if mpmd_nodes:
            run_settings = self._build_run_settings("python", exe_args_mpmd, **kwargs)
            node = DBNode(db_node_name, self.path, run_settings, [port])
            node._mpmd = True
            node._shard_ids = range(db_nodes)
            self.entities.append(node)

        self.ports = [port]

    @property
    def num_shards(self):
        return self.db_nodes

    def _fill_reserved(self):
        """Fill the reserved batch and run arguments dictionaries"""
        self._reserved_run_args[MpirunSettings] = [
            "np",
            "N",
            "c",
            "output-filename",
            "n",
            "wdir",
            "wd",
            "host",
        ]
        self._reserved_run_args[SrunSettings] = [
            "nodes",
            "N",
            "ntasks",
            "n",
            "ntasks-per-node",
            "output",
            "o",
            "error",
            "e",
            "job-name",
            "J",
            "jobid",
            "multi-prog",
            "w",
            "chdir",
            "D",
        ]
        self._reserved_run_args[AprunSettings] = [
            "pes",
            "n",
            "pes-per-node",
            "N",
            "l",
            "pes-per-numa-node",
            "S",
            "wdir",
        ]
        self._reserved_batch_args[SbatchSettings] = [
            "nodes",
            "N",
            "ntasks",
            "n",
            "ntasks-per-node",
            "output",
            "o",
            "error",
            "e",
            "job-name",
            "J",
            "jobid",
            "multi-prog",
            "w",
            "chdir",
            "D",
        ]
        self._reserved_batch_args[CobaltBatchSettings] = [
            "cwd",
            "error",
            "e",
            "output",
            "o",
            "outputprefix",
            "N",
            "l",
            "jobname",
        ]
        self._reserved_batch_args[QsubBatchSettings] = ["e", "o", "N", "l"]