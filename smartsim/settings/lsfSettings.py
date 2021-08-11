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

from pprint import pformat

from ..error import SSConfigError
from ..utils.helpers import init_default
from .settings import BatchSettings, RunSettings


class JsrunSettings(RunSettings):
    def __init__(self, exe, exe_args=None, run_args=None, env_vars=None):
        """Settings to run job with ``jsrun`` command

        ``JsrunSettings`` can be used for both the `lsf` launcher.

        :param exe: executable
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: str | list[str], optional
        :param run_args: arguments for run command, defaults to None
        :type run_args: dict[str, str], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        super().__init__(
            exe, exe_args, run_command="jsrun", run_args=run_args, env_vars=env_vars
        )

        # Parameters needed for MPMD run
        self.erf_sets = {"host": "*", "cpu": "*", "ranks": "1"}
        self.mpmd_preamble_lines = []
        self.mpmd = []
        self.individual_suffix = None

    def set_num_rs(self, num_rs):
        """Set the number of resource sets to use

        This sets ``--nrs``.

        :param num_rs: Number of resource sets or `ALL_HOSTS`
        :type num_rs: int or str
        """
        if isinstance(num_rs, str):
            self.run_args["nrs"] = num_rs
        else:
            self.run_args["nrs"] = int(num_rs)

    def set_cpus_per_rs(self, num_cpus):
        """Set the number of cpus to use per resource set

        This sets ``--cpu_per_rs``

        :param num_cpus: number of cpus to use per resource set or ALL_CPUS
        :type num_cpus: int or str
        """
        if isinstance(num_cpus, str):
            self.run_args["cpu_per_rs"] = num_cpus
        else:
            self.run_args["cpu_per_rs"] = int(num_cpus)

    def set_gpus_per_rs(self, num_gpus):
        """Set the number of gpus to use per resource set

        This sets ``--gpu_per_rs``

        :param num_cpus: number of gpus to use per resource set or ALL_GPUS
        :type num_gpus: int or str
        """
        if isinstance(num_gpus, str):
            self.run_args["gpu_per_rs"] = num_gpus
        else:
            self.run_args["gpu_per_rs"] = int(num_gpus)

    def set_rs_per_host(self, num_rs):
        """Set the number of resource sets to use per host

        This sets ``--rs_per_host``

        :param num_rs: number of resource sets to use per host
        :type num_rs: int
        """
        self.run_args["rs_per_host"] = int(num_rs)

    def set_tasks(self, num_tasks):
        """Set the number of tasks for this job

        This sets ``--np``

        :param num_tasks: number of tasks
        :type num_tasks: int
        """
        self.run_args["np"] = int(num_tasks)

    def set_tasks_per_rs(self, num_tprs):
        """Set the number of tasks per resource set

        This sets ``--tasks_per_rs``

        :param num_tpn: number of tasks per resource set
        :type num_tpn: int
        """
        self.run_args["tasks_per_rs"] = int(num_tprs)

    def set_binding(self, binding):
        """Set binding

        This sets ``--bind``

        :param binding: Binding, e.g. `packed:21`
        :type binding: str
        """
        self.run_args["bind"] = binding

    def make_mpmd(self, jsrun_settings=None):
        """Make step an MPMD (or SPMD) job.

        This method will activate job execution through an ERF file.

        Optionally, this method adds an instance of ``JsrunSettings`` to
        the list of settings to be launched in the same ERF file.

        :param aprun_settings: ``JsrunSettings`` instance, defaults to None
        :type aprun_settings: JsrunSettings, optional
        """
        if len(self.mpmd) == 0:
            self.mpmd.append(self)
        if jsrun_settings:
            self.mpmd.append(jsrun_settings)

    def set_mpmd_preamble(self, preamble_lines):
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

    def set_erf_sets(self, erf_sets):
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
        self.erf_sets = erf_sets

    def format_env_vars(self):
        """Format environment variables. Each variable needs
        to be passed with ``--env``. If a variable is set to ``None``,
        its value is propagated from the current environment.

        :returns: formatted string to export variables
        :rtype: str
        """
        format_str = ""
        for k, v in self.env_vars.items():
            if v:
                format_str += f"-E {k}={v} "
            else:
                format_str += f"-E {k} "
        return format_str.rstrip(" ")

    def set_individual_output(self, suffix=None):
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

    def format_run_args(self):
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

    def __str__(self):
        string = super().__str__()
        if self.mpmd:
            string += "\nERF settings: " + pformat(self.erf_sets)
        return string


class BsubBatchSettings(BatchSettings):
    def __init__(
        self,
        nodes=None,
        time=None,
        project=None,
        batch_args=None,
        smts=None,
        **kwargs,
    ):
        """Specify ``bsub`` batch parameters for a job

        :param nodes: number of nodes for batch, defaults to None
        :type nodes: int, optional
        :param time: walltime for batch job in format hh:mm, defaults to None
        :type time: str, optional
        :param project: project for batch launch, defaults to None
        :type project: str, optional
        :param batch_args: overrides for LSF batch arguments, defaults to None
        :type batch_args: dict[str, str], optional
        :param smts: SMTs, defaults to None
        :type smts: int, optional
        """
        super().__init__("bsub", batch_args=batch_args)
        if nodes:
            self.set_nodes(nodes)
        self.set_walltime(time)
        self.set_project(project)
        if smts:
            self.set_smts(smts)
        else:
            self.smts = None
        self.expert_mode = False
        self.easy_settings = ["ln_slots", "ln_mem", "cn_cu", "nnodes"]

    def set_walltime(self, time):
        """Set the walltime

        This sets ``-W``.

        :param time: Time in hh:mm format, e.g. "10:00" for 10 hours
        :type time: str
        """
        self.walltime = time

    def set_smts(self, smts):
        """Set SMTs

        This sets ``-alloc_flags``. If the user sets
        SMT explicitly through ``-alloc_flags``, then that
        takes precedence.

        :param smts: SMT (e.g on Summit: 1, 2, or 4)
        :type smts: int
        """
        self.smts = int(smts)

    def set_project(self, project):
        """Set the project

        This sets ``-P``.

        :param time: project name
        :type time: str
        """
        self.project = project

    def set_nodes(self, num_nodes):
        """Set the number of nodes for this batch job

        This sets ``-nnodes``.

        :param num_nodes: number of nodes
        :type num_nodes: int
        """
        self.batch_args["nnodes"] = int(num_nodes)

    def set_expert_mode_req(self, res_req, slots):
        """Set allocation for expert mode. This
        will activate expert mode (``-csm``) and
        disregard all other allocation options.

        This sets ``-csm -n slots -R res_req``
        """
        self.expert_mode = True
        self.batch_args["csm"] = "y"
        self.batch_args["R"] = res_req
        self.batch_args["n"] = slots

    def set_hostlist(self, host_list):
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :type host_list: str | list[str]
        :raises TypeError: if not str or list of str
        """
        if isinstance(host_list, str):
            host_list = [host_list.strip()]
        if not isinstance(host_list, list):
            raise TypeError("host_list argument must be a list of strings")
        if not all([isinstance(host, str) for host in host_list]):
            raise TypeError("host_list argument must be list of strings")
        self.batch_args["m"] = '"' + " ".join(host_list) + '"'

    def set_tasks(self, num_tasks):
        """Set the number of tasks for this job

        This sets ``-n``

        :param num_tasks: number of tasks
        :type num_tasks: int
        """
        self.batch_args["n"] = int(num_tasks)

    def _format_alloc_flags(self):
        """Format ``alloc_flags`` checking if user already
        set it. Currently only adds SMT flag if missing
        and ``self.smts`` is set.
        """

        if self.smts:
            if not "alloc_flags" in self.batch_args.keys():
                self.batch_args["alloc_flags"] = f"smt{self.smts}"
            else:
                # Check if smt is in the flag, otherwise add it
                flags = self.batch_args["alloc_flags"].strip('"').split()
                if not any([flag.startswith("smt") for flag in flags]):
                    flags.append(f"smt{self.smts}")
                    self.batch_args["alloc_flags"] = " ".join(flags)

        # Check if alloc_flags has to be enclosed in quotes
        if "alloc_flags" in self.batch_args.keys():
            flags = self.batch_args["alloc_flags"].strip('"').split()
            if len(flags) > 1:
                self.batch_args["alloc_flags"] = '"' + " ".join(flags) + '"'

    def format_batch_args(self):
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
