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

from __future__ import annotations

import copy
import typing as t
from pprint import pformat

from ..error import SSUnsupportedError
from ..log import get_logger
from .base import BatchSettings, RunSettings

logger = get_logger(__name__)


class JsrunSettings(RunSettings):
    def __init__(
        self,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.List[str]]] = None,
        run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **_kwargs: t.Any,
    ) -> None:
        """Settings to run job with ``jsrun`` command

        ``JsrunSettings`` should only be used on LSF-based systems.

        :param exe: executable
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: str | list[str], optional
        :param run_args: arguments for run command, defaults to None
        :type run_args: dict[str, t.Union[int, str, float, None]], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(
            exe,
            exe_args,
            run_command="jsrun",
            run_args=run_args,
            env_vars=env_vars,
        )

        # Parameters needed for MPMD run
        self.erf_sets = {"host": "*", "cpu": "*", "ranks": "1"}
        self.mpmd_preamble_lines: t.List[str] = []
        self.mpmd: t.List[RunSettings] = []
        self.individual_suffix = ""

    reserved_run_args = {"chdir", "h"}

    def set_num_rs(self, num_rs: t.Union[str, int]) -> None:
        """Set the number of resource sets to use

        This sets ``--nrs``.

        :param num_rs: Number of resource sets or `ALL_HOSTS`
        :type num_rs: int or str
        """
        if isinstance(num_rs, str):
            self.run_args["nrs"] = num_rs
        else:
            self.run_args["nrs"] = int(num_rs)

    def set_cpus_per_rs(self, cpus_per_rs: int) -> None:
        """Set the number of cpus to use per resource set

        This sets ``--cpu_per_rs``

        :param cpus_per_rs: number of cpus to use per resource set or ALL_CPUS
        :type cpus_per_rs: int or str
        """
        if self.colocated_db_settings:
            db_cpus = int(self.colocated_db_settings.get("db_cpus", 0))
            if not db_cpus:
                raise ValueError(
                    "db_cpus must be configured on colocated_db_settings"
                )

            if cpus_per_rs < db_cpus:
                raise ValueError(
                    f"Cannot set cpus_per_rs ({cpus_per_rs}) to less than "
                    + f"db_cpus ({db_cpus})"
                )
        if isinstance(cpus_per_rs, str):
            self.run_args["cpu_per_rs"] = cpus_per_rs
        else:
            self.run_args["cpu_per_rs"] = int(cpus_per_rs)

    def set_gpus_per_rs(self, gpus_per_rs: int) -> None:
        """Set the number of gpus to use per resource set

        This sets ``--gpu_per_rs``

        :param gpus_per_rs: number of gpus to use per resource set or ALL_GPUS
        :type gpus_per_rs: int or str
        """
        if isinstance(gpus_per_rs, str):
            self.run_args["gpu_per_rs"] = gpus_per_rs
        else:
            self.run_args["gpu_per_rs"] = int(gpus_per_rs)

    def set_rs_per_host(self, rs_per_host: int) -> None:
        """Set the number of resource sets to use per host

        This sets ``--rs_per_host``

        :param rs_per_host: number of resource sets to use per host
        :type rs_per_host: int
        """
        self.run_args["rs_per_host"] = int(rs_per_host)

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        This sets ``--np``

        :param tasks: number of tasks
        :type tasks: int
        """
        self.run_args["np"] = int(tasks)

    def set_tasks_per_rs(self, tasks_per_rs: int) -> None:
        """Set the number of tasks per resource set

        This sets ``--tasks_per_rs``

        :param tasks_per_rs: number of tasks per resource set
        :type tasks_per_rs: int
        """
        self.run_args["tasks_per_rs"] = int(tasks_per_rs)

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks per resource set.

        This function is an alias for `set_tasks_per_rs`.

        :param tasks_per_node: number of tasks per resource set
        :type tasks_per_node: int
        """
        self.set_tasks_per_rs(int(tasks_per_node))

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of cpus per tasks.

        This function is an alias for `set_cpus_per_rs`.

        :param cpus_per_task: number of cpus per resource set
        :type cpus_per_task: int
        """
        self.set_cpus_per_rs(int(cpus_per_task))

    def set_memory_per_rs(self, memory_per_rs: int) -> None:
        """Specify the number of megabytes of memory to assign to a resource set

        This sets ``--memory_per_rs``

        :param memory_per_rs: Number of megabytes per rs
        :type memory_per_rs: int
        """
        self.run_args["memory_per_rs"] = int(memory_per_rs)

    def set_memory_per_node(self, memory_per_node: int) -> None:
        """Specify the number of megabytes of memory to assign to a resource set

        Alias for `set_memory_per_rs`.

        :param memory_per_node: Number of megabytes per rs
        :type memory_per_node: int
        """
        self.set_memory_per_rs(int(memory_per_node))

    def set_binding(self, binding: str) -> None:
        """Set binding

        This sets ``--bind``

        :param binding: Binding, e.g. `packed:21`
        :type binding: str
        """
        self.run_args["bind"] = binding

    def make_mpmd(self, settings: RunSettings) -> None:
        """Make step an MPMD (or SPMD) job.

        This method will activate job execution through an ERF file.

        Optionally, this method adds an instance of ``JsrunSettings`` to
        the list of settings to be launched in the same ERF file.

        :param settings: ``JsrunSettings`` instance
        :type settings: JsrunSettings, optional
        """
        if self.colocated_db_settings:
            raise SSUnsupportedError(
                "Colocated models cannot be run as a mpmd workload"
            )

        self.mpmd.append(settings)

    def set_mpmd_preamble(self, preamble_lines: t.List[str]) -> None:
        """Set preamble used in ERF file. Typical lines include
        `oversubscribe-cpu : allow` or `overlapping-rs : allow`.
        Can be used to set `launch_distribution`. If it is not present,
        it will be inferred from the settings, or set to `packed` by
        default.

        :param preamble_lines: lines to put at the beginning of the ERF
                               file.
        :type preamble_lines: list[str]
        """
        self.mpmd_preamble_lines = preamble_lines

    def set_erf_sets(self, erf_sets: t.Dict[str, str]) -> None:
        """Set resource sets used for ERF (SPMD or MPMD) steps.

        ``erf_sets`` is a dictionary used to fill the ERF
        line representing these settings, e.g.
        `{"host": "1", "cpu": "{0:21}, {21:21}", "gpu": "*"}`
        can be used to specify rank (or rank_count), hosts, cpus, gpus,
        and memory.
        The key `rank` is used to give specific ranks, as in
        `{"rank": "1, 2, 5"}`, while the key `rank_count` is used to specify
        the count only, as in `{"rank_count": "3"}`. If both are specified,
        only `rank` is used.

        :param hosts: dictionary of resources
        :type hosts: dict[str,str]
        """
        self.erf_sets = copy.deepcopy(erf_sets)

    def format_env_vars(self) -> t.List[str]:
        """Format environment variables. Each variable needs
        to be passed with ``--env``. If a variable is set to ``None``,
        its value is propagated from the current environment.

        :returns: formatted list of strings to export variables
        :rtype: list[str]
        """
        format_str = []
        for k, v in self.env_vars.items():
            if v:
                format_str += ["-E", f"{k}={v}"]
            else:
                format_str += ["-E", f"{k}"]
        return format_str

    def set_individual_output(self, suffix: t.Optional[str] = None) -> None:
        """Set individual std output.

        This sets ``--stdio_mode individual``
        and inserts the suffix into the output name. The resulting
        output name will be ``self.name + suffix + .out``.

        :param suffix: Optional suffix to add to output file names,
                       it can contain `%j`, `%h`, `%p`, or `%t`,
                       as specified by `jsrun` options.
        :type suffix: str, optional

        """
        self.run_args["stdio_mode"] = "individual"
        if suffix:
            self.individual_suffix = suffix

    def format_run_args(self) -> t.List[str]:
        """Return a list of LSF formatted run arguments

        :return: list of LSF arguments for these settings
        :rtype: list[str]
        """
        # args launcher uses
        args = []
        restricted = ["chdir", "h", "stdio_stdout", "o", "stdio_stderr", "k"]
        if self.mpmd or "erf_input" in self.run_args.keys():
            restricted.extend(
                [
                    "tasks_per_rs",
                    "a",
                    "np",
                    "p",
                    "cpu_per_rs",
                    "c",
                    "gpu_per_rs",
                    "g",
                    "latency_priority",
                    "l",
                    "memory_per_rs",
                    "m",
                    "nrs",
                    "n",
                    "rs_per_host",
                    "r",
                    "rs_per_socket",
                    "K",
                    "appfile",
                    "f",
                    "allocate_only",
                    "A",
                    "launch_node_task",
                    "H",
                    "use_reservation",
                    "J",
                    "use_resources",
                    "bind",
                    "b",
                    "launch_distribution",
                    "d",
                ]
            )

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

    def __str__(self) -> str:
        string = super().__str__()
        if self.mpmd:
            string += "\nERF settings: " + pformat(self.erf_sets)
        return string

    def _prep_colocated_db(self, db_cpus: int) -> None:
        cpus_per_flag_set = False
        for cpu_per_rs_flag in ["cpu_per_rs", "c"]:
            if run_arg_value := self.run_args.get(cpu_per_rs_flag, 0):
                cpus_per_flag_set = True
                cpu_per_rs = int(run_arg_value)
                if cpu_per_rs < db_cpus:
                    msg = (
                        f"{cpu_per_rs_flag} flag was set to {cpu_per_rs}, but "
                        f"colocated DB requires {db_cpus} CPUs per RS. Automatically "
                        f"setting {cpu_per_rs_flag} flag to {db_cpus}"
                    )
                    logger.info(msg)
                    self.run_args[cpu_per_rs_flag] = db_cpus
        if not cpus_per_flag_set:
            msg = f"Colocated DB requires {db_cpus} CPUs per RS. Automatically setting "
            msg += f"--cpus_per_rs=={db_cpus}"
            logger.info(msg)
            self.set_cpus_per_rs(db_cpus)

        rs_per_host_set = False
        for rs_per_host_flag in ["rs_per_host", "r"]:
            if rs_per_host_flag in self.run_args:
                rs_per_host_set = True
                rs_per_host = self.run_args[rs_per_host_flag]
                if rs_per_host != 1:
                    msg = f"{rs_per_host_flag} flag was set to {rs_per_host}, "
                    msg += (
                        "but colocated DB requires running ONE resource set per host. "
                    )
                    msg += f"Automatically setting {rs_per_host_flag} flag to 1"
                    logger.info(msg)
                    self.run_args[rs_per_host_flag] = "1"
        if not rs_per_host_set:
            msg = "Colocated DB requires one resource set per host. "
            msg += " Automatically setting --rs_per_host==1"
            logger.info(msg)
            self.set_rs_per_host(1)


class BsubBatchSettings(BatchSettings):
    def __init__(
        self,
        nodes: t.Optional[int] = None,
        time: t.Optional[str] = None,
        project: t.Optional[str] = None,
        batch_args: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        smts: int = 0,
        **kwargs: t.Any,
    ) -> None:
        """Specify ``bsub`` batch parameters for a job

        :param nodes: number of nodes for batch, defaults to None
        :type nodes: int, optional
        :param time: walltime for batch job in format hh:mm, defaults to None
        :type time: str, optional
        :param project: project for batch launch, defaults to None
        :type project: str, optional
        :param batch_args: overrides for LSF batch arguments, defaults to None
        :type batch_args: dict[str, str], optional
        :param smts: SMTs, defaults to 0
        :type smts: int, optional
        """
        self.project: t.Optional[str] = None

        if project:
            kwargs.pop("account", None)
        else:
            project = kwargs.pop("account", None)

        super().__init__(
            "bsub",
            batch_args=batch_args,
            nodes=nodes,
            account=project,
            time=time,
            **kwargs,
        )

        self.smts = 0
        if smts:
            self.set_smts(smts)

        self.expert_mode = False
        self.easy_settings = ["ln_slots", "ln_mem", "cn_cu", "nnodes"]

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime

        This sets ``-W``.

        :param walltime: Time in hh:mm format, e.g. "10:00" for 10 hours,
                         if time is supplied in hh:mm:ss format, seconds
                         will be ignored and walltime will be set as ``hh:mm``
        :type walltime: str
        """
        # For compatibility with other launchers, as explained in docstring
        if walltime:
            if len(walltime.split(":")) > 2:
                walltime = ":".join(walltime.split(":")[:2])
        self.walltime = walltime

    def set_smts(self, smts: int) -> None:
        """Set SMTs

        This sets ``-alloc_flags``. If the user sets
        SMT explicitly through ``-alloc_flags``, then that
        takes precedence.

        :param smts: SMT (e.g on Summit: 1, 2, or 4)
        :type smts: int
        """
        self.smts = smts

    def set_project(self, project: str) -> None:
        """Set the project

        This sets ``-P``.

        :param time: project name
        :type time: str
        """
        if project:
            self.project = project

    def set_account(self, account: str) -> None:
        """Set the project

        this function is an alias for `set_project`.

        :param account: project name
        :type account: str
        """
        self.set_project(account)

    def set_nodes(self, num_nodes: int) -> None:
        """Set the number of nodes for this batch job

        This sets ``-nnodes``.

        :param nodes: number of nodes
        :type nodes: int
        """
        if num_nodes:
            self.batch_args["nnodes"] = str(int(num_nodes))

    def set_expert_mode_req(self, res_req: str, slots: int) -> None:
        """Set allocation for expert mode. This
        will activate expert mode (``-csm``) and
        disregard all other allocation options.

        This sets ``-csm -n slots -R res_req``
        """
        self.expert_mode = True
        self.batch_args["csm"] = "y"
        self.batch_args["R"] = res_req
        self.batch_args["n"] = str(slots)

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
        self.batch_args["m"] = '"' + " ".join(host_list) + '"'

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        This sets ``-n``

        :param tasks: number of tasks
        :type tasks: int
        """
        self.batch_args["n"] = str(int(tasks))

    def set_queue(self, queue: str) -> None:
        """Set the queue for this job

        :param queue: The queue to submit the job on
        :type queue: str
        """
        if queue:
            self.batch_args["q"] = queue

    def _format_alloc_flags(self) -> None:
        """Format ``alloc_flags`` checking if user already
        set it. Currently only adds SMT flag if missing
        and ``self.smts`` is set.
        """

        if self.smts:
            if "alloc_flags" not in self.batch_args.keys():
                self.batch_args["alloc_flags"] = f"smt{self.smts}"
            else:
                # Check if smt is in the flag, otherwise add it
                flags: t.List[str] = []
                if flags_arg := self.batch_args.get("alloc_flags", ""):
                    flags = flags_arg.strip('"').split()
                if not any(flag.startswith("smt") for flag in flags):
                    flags.append(f"smt{self.smts}")
                    self.batch_args["alloc_flags"] = " ".join(flags)

        # Check if alloc_flags has to be enclosed in quotes
        if "alloc_flags" in self.batch_args.keys():
            flags = []
            if flags_arg := self.batch_args.get("alloc_flags", ""):
                flags = flags_arg.strip('"').split()
            if len(flags) > 1:
                self.batch_args["alloc_flags"] = '"' + " ".join(flags) + '"'

    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview

        :return: list of batch arguments for Qsub
        :rtype: list[str]
        """
        opts = []

        self._format_alloc_flags()

        for opt, value in self.batch_args.items():
            if self.expert_mode and opt in self.easy_settings:
                continue

            prefix = "-"  # LSF only uses single dashses

            if not value:
                opts += [prefix + opt]
            else:
                opts += [" ".join((prefix + opt, str(value)))]

        return opts
