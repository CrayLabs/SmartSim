# BSD 2-Clause License #
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

import copy
import typing as t

from smartsim.settings.containers import Container

from .._core.utils.helpers import expand_exe_path, fmt_dict, is_valid_cmd
from ..entity.dbobject import DBModel, DBScript
from ..log import get_logger

logger = get_logger(__name__)

# fmt: off
class SettingsBase:
    ...
# fmt: on


# pylint: disable=too-many-public-methods
class RunSettings(SettingsBase):
    # pylint: disable=unused-argument

    def __init__(
        self,
        run_command: str = "",
        run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        container: t.Optional[Container] = None,
        **_kwargs: t.Any,
    ) -> None:
        """Run parameters for a `Aapplication``

        The base ``RunSettings`` class should only be used with the `local`
        launcher on single node, workstations, or laptops.

        If no ``run_command`` is specified, the executable will be launched
        locally.

        ``run_args`` passed as a dict will be interpreted literally for
        local ``RunSettings`` and added directly to the ``run_command``
        e.g. run_args = {"-np": 2} will be "-np 2"

        Example initialization

        .. highlight:: python
        .. code-block:: python

            rs = RunSettings("echo", "hello", "mpirun", run_args={"-np": "2"})

        :param run_command: launch binary (e.g. "srun")
        :param run_args: arguments for run command (e.g. `-np` for `mpiexec`)
        :param env_vars: environment vars to launch job with
        :param container: container type for workload (e.g. "singularity")
        """
        # Do not expand executable if running within a container
        self.run_args = run_args or {}
        self.env_vars = env_vars or {}
        self.container = container
        self._run_command = run_command
        self.in_batch = False
        self.colocated_db_settings: t.Optional[
            t.Dict[
                str,
                t.Union[
                    bool,
                    int,
                    str,
                    None,
                    t.List[str],
                    t.Iterable[t.Union[int, t.Iterable[int]]],
                    t.List[DBModel],
                    t.List[DBScript],
                    t.Dict[str, t.Union[int, None]],
                    t.Dict[str, str],
                ],
            ]
        ] = None

    @property
    def run_args(self) -> t.Dict[str, t.Union[int, str, float, None]]:
        """Return an immutable list of attached run arguments.

        :returns: attached run arguments
        """
        return self._run_args

    @run_args.setter
    def run_args(self, value: t.Dict[str, t.Union[int, str, float, None]]) -> None:
        """Set the run arguments.

        :param value: run arguments
        """
        self._run_args = copy.deepcopy(value)

    @property
    def env_vars(self) -> t.Dict[str, t.Optional[str]]:
        """Return an immutable list of attached environment variables.

        :returns: attached environment variables
        """
        return self._env_vars

    @env_vars.setter
    def env_vars(self, value: t.Dict[str, t.Optional[str]]) -> None:
        """Set the environment variables.

        :param value: environment variables
        """
        self._env_vars = copy.deepcopy(value)

    # To be overwritten by subclasses. Set of reserved args a user cannot change
    reserved_run_args = set()  # type: set[str]

    def set_nodes(self, nodes: int) -> None:
        """Set the number of nodes

        :param nodes: number of nodes to run with
        """
        logger.warning(
            (
                "Node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks to launch

        :param tasks: number of tasks to launch
        """
        logger.warning(
            (
                "Task specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks per node

        :param tasks_per_node: number of tasks to launch per node
        """
        logger.warning(
            (
                "Task per node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_task_map(self, task_mapping: str) -> None:
        """Set a task mapping

        :param task_mapping: task mapping
        """
        logger.warning(
            (
                "Task mapping specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of cpus per task

        :param cpus_per_task: number of cpus per task
        """
        logger.warning(
            (
                "CPU per node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        """
        logger.warning(
            (
                "Hostlist specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_hostlist_from_file(self, file_path: str) -> None:
        """Use the contents of a file to specify the hostlist for this job

        :param file_path: Path to the hostlist file
        """
        logger.warning(
            (
                "Hostlist from file specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_excluded_hosts(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify a list of hosts to exclude for launching this job

        :param host_list: hosts to exclude
        """
        logger.warning(
            (
                "Excluded host list specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_cpu_bindings(self, bindings: t.Union[int, t.List[int]]) -> None:
        """Set the cores to which MPI processes are bound

        :param bindings: List specifing the cores to which MPI processes are bound
        """
        logger.warning(
            (
                "CPU binding specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_memory_per_node(self, memory_per_node: int) -> None:
        """Set the amount of memory required per node in megabytes

        :param memory_per_node: Number of megabytes per node
        """
        logger.warning(
            (
                "Memory per node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_verbose_launch(self, verbose: bool) -> None:
        """Set the job to run in verbose mode

        :param verbose: Whether the job should be run verbosely
        """
        logger.warning(
            (
                "Verbose specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_quiet_launch(self, quiet: bool) -> None:
        """Set the job to run in quiet mode

        :param quiet: Whether the job should be run quietly
        """
        logger.warning(
            (
                "Quiet specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_broadcast(self, dest_path: t.Optional[str] = None) -> None:
        """Copy executable file to allocated compute nodes

        :param dest_path: Path to copy an executable file
        """
        logger.warning(
            (
                "Broadcast specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_time(self, hours: int = 0, minutes: int = 0, seconds: int = 0) -> None:
        """Automatically format and set wall time

        :param hours: number of hours to run job
        :param minutes: number of minutes to run job
        :param seconds: number of seconds to run job
        """
        return self.set_walltime(
            self._fmt_walltime(int(hours), int(minutes), int(seconds))
        )

    def set_node_feature(self, feature_list: t.Union[str, t.List[str]]) -> None:
        """Specify the node feature for this job

        :param feature_list: node feature to launch on
        """
        logger.warning(
            (
                "Feature specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    @staticmethod
    def _fmt_walltime(hours: int, minutes: int, seconds: int) -> str:
        """Convert hours, minutes, and seconds into valid walltime format

        By defualt the formatted wall time is the total number of seconds.

        :param hours: number of hours to run job
        :param minutes: number of minutes to run job
        :param seconds: number of seconds to run job
        :returns: Formatted walltime
        """
        time_ = hours * 3600
        time_ += minutes * 60
        time_ += seconds
        return str(time_)

    def set_walltime(self, walltime: str) -> None:
        """Set the formatted walltime

        :param walltime: Time in format required by launcher``
        """
        logger.warning(
            (
                "Walltime specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_binding(self, binding: str) -> None:
        """Set binding

        :param binding: Binding
        """
        logger.warning(
            (
                "binding specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_mpmd_preamble(self, preamble_lines: t.List[str]) -> None:
        """Set preamble to a file to make a job MPMD

        :param preamble_lines: lines to put at the beginning of a file.
        """
        logger.warning(
            (
                "MPMD preamble specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def make_mpmd(self, settings: RunSettings) -> None:
        """Make job an MPMD job

        :param settings: ``RunSettings`` instance
        """
        logger.warning(
            (
                "Make MPMD specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    @property
    def run_command(self) -> t.Optional[str]:
        """Return the launch binary used to launch the executable

        Attempt to expand the path to the executable if possible

        :returns: launch binary e.g. mpiexec
        """
        cmd = self._run_command

        if cmd:
            if is_valid_cmd(cmd):
                # command is valid and will be expanded
                return expand_exe_path(cmd)
            # command is not valid, so return it as is
            # it may be on the compute nodes but not local machine
            return cmd
        # run without run command
        return None

    def update_env(self, env_vars: t.Dict[str, t.Union[str, int, float, bool]]) -> None:
        """Update the job environment variables

        To fully inherit the current user environment, add the
        workload-manager-specific flag to the launch command through the
        :meth:`add_exe_args` method. For example, ``--export=ALL`` for
        slurm, or ``-V`` for PBS/aprun.


        :param env_vars: environment variables to update or add
        :raises TypeError: if env_vars values cannot be coerced to strings
        """
        val_types = (str, int, float, bool)
        # Coerce env_vars values to str as a convenience to user
        for env, val in env_vars.items():
            if not isinstance(val, val_types):
                raise TypeError(
                    f"env_vars[{env}] was of type {type(val)}, not {val_types}"
                )

            self.env_vars[env] = str(val)

    def set(
        self, arg: str, value: t.Optional[str] = None, condition: bool = True
    ) -> None:
        """Allows users to set individual run arguments.

        A method that allows users to set run arguments after object
        instantiation. Does basic formatting such as stripping leading dashes.
        If the argument has been set previously, this method will log warning
        but ultimately comply.

        Conditional expressions may be passed to the conditional parameter. If the
        expression evaluates to True, the argument will be set. In not an info
        message is logged and no further operation is performed.

        Basic Usage

        .. highlight:: python
        .. code-block:: python

            rs = RunSettings("python")
            rs.set("an-arg", "a-val")
            rs.set("a-flag")
            rs.format_run_args()  # returns ["an-arg", "a-val", "a-flag", "None"]

        Slurm Example with Conditional Setting

        .. highlight:: python
        .. code-block:: python

            import socket

            rs = SrunSettings("echo", "hello")
            rs.set_tasks(1)
            rs.set("exclusive")

            # Only set this argument if condition param evals True
            # Otherwise log and NOP
            rs.set("partition", "debug",
                   condition=socket.gethostname()=="testing-system")

            rs.format_run_args()
            # returns ["exclusive", "None", "partition", "debug"] iff
              socket.gethostname()=="testing-system"
            # otherwise returns ["exclusive", "None"]

        :param arg: name of the argument
        :param value: value of the argument
        :param conditon: set the argument if condition evaluates to True
        """
        if not isinstance(arg, str):
            raise TypeError("Argument name should be of type str")
        if value is not None and not isinstance(value, str):
            raise TypeError("Argument value should be of type str or None")
        arg = arg.strip().lstrip("-")

        if not condition:
            logger.info(f"Could not set argument '{arg}': condition not met")
            return
        if arg in self.reserved_run_args:
            logger.warning(
                (
                    f"Could not set argument '{arg}': "
                    f"it is a reserved arguement of '{type(self).__name__}'"
                )
            )
            return

        if arg in self.run_args and value != self.run_args[arg]:
            logger.warning(f"Overwritting argument '{arg}' with value '{value}'")
        self.run_args[arg] = value

    def format_run_args(self) -> t.List[str]:
        """Return formatted run arguments

        For ``RunSettings``, the run arguments are passed
        literally with no formatting.

        :return: list run arguments for these settings
        """
        formatted = []
        for arg, value in self.run_args.items():
            formatted.append(arg)
            formatted.append(str(value))
        return formatted

    def format_env_vars(self) -> t.List[str]:
        """Build environment variable string

        :returns: formatted list of strings to export variables
        """
        formatted = []
        for key, val in self.env_vars.items():
            if val is None:
                formatted.append(f"{key}=")
            else:
                formatted.append(f"{key}={val}")
        return formatted

    def __str__(self) -> str:  # pragma: no-cover
        string = ""
        if self.run_command:
            string += f"\nRun Command: {self.run_command}"
        if self.run_args:
            string += f"\nRun Arguments:\n{fmt_dict(self.run_args)}"
        if self.colocated_db_settings:
            string += "\nCo-located Database: True"
        return string


class BatchSettings(SettingsBase):
    def __init__(
        self,
        batch_cmd: str,
        batch_args: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ) -> None:
        self._batch_cmd = batch_cmd
        self.batch_args = batch_args or {}
        self._preamble: t.List[str] = []
        self.set_nodes(kwargs.get("nodes", None))
        self.set_walltime(kwargs.get("time", None))
        self.set_queue(kwargs.get("queue", None))
        self.set_account(kwargs.get("account", None))

    @property
    def batch_cmd(self) -> str:
        """Return the batch command

        Tests to see if we can expand the batch command
        path. If we can, then returns the expanded batch
        command. If we cannot, returns the batch command as is.

        :returns: batch command
        """
        if is_valid_cmd(self._batch_cmd):
            return expand_exe_path(self._batch_cmd)

        return self._batch_cmd

    @property
    def batch_args(self) -> t.Dict[str, t.Optional[str]]:
        """Retrieve attached batch arguments

        :returns: attached batch arguments
        """
        return self._batch_args

    @batch_args.setter
    def batch_args(self, value: t.Dict[str, t.Optional[str]]) -> None:
        """Attach batch arguments

        :param value: dictionary of batch arguments
        """
        self._batch_args = copy.deepcopy(value) if value else {}

    def set_nodes(self, num_nodes: int) -> None:
        raise NotImplementedError

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        raise NotImplementedError

    def set_queue(self, queue: str) -> None:
        raise NotImplementedError

    def set_walltime(self, walltime: str) -> None:
        raise NotImplementedError

    def set_account(self, account: str) -> None:
        raise NotImplementedError

    def format_batch_args(self) -> t.List[str]:
        raise NotImplementedError

    def set_batch_command(self, command: str) -> None:
        """Set the command used to launch the batch e.g. ``sbatch``

        :param command: batch command
        """
        self._batch_cmd = command

    def add_preamble(self, lines: t.List[str]) -> None:
        """Add lines to the batch file preamble. The lines are just
        written (unmodified) at the beginning of the batch file
        (after the WLM directives) and can be used to e.g.
        start virtual environments before running the executables.

        :param line: lines to add to preamble.
        """
        if isinstance(lines, str):
            self._preamble += [lines]
        elif isinstance(lines, list):
            self._preamble += lines
        else:
            raise TypeError("Expected str or List[str] for lines argument")

    @property
    def preamble(self) -> t.Iterable[str]:
        """Return an iterable of preamble clauses to be prepended to the batch file

        :return: attached preamble clauses
        """
        return (clause for clause in self._preamble)

    def __str__(self) -> str:  # pragma: no-cover
        string = f"Batch Command: {self._batch_cmd}"
        if self.batch_args:
            string += f"\nBatch arguments:\n{fmt_dict(self.batch_args)}"
        return string
