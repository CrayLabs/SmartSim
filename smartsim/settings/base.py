# BSD 2-Clause License #
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

from .._core.utils.helpers import expand_exe_path, fmt_dict, init_default, is_valid_cmd
from ..log import get_logger

logger = get_logger(__name__)


class RunSettings:
    def __init__(
        self,
        exe,
        exe_args=None,
        run_command="",
        run_args=None,
        env_vars=None,
        container=None,
        **kwargs,
    ):
        """Run parameters for a ``Model``

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

        :param exe: executable to run
        :type exe: str
        :param exe_args: executable arguments, defaults to None
        :type exe_args: str | list[str], optional
        :param run_command: launch binary (e.g. "srun"), defaults to empty str
        :type run_command: str, optional
        :param run_args: arguments for run command (e.g. `-np` for `mpiexec`), defaults to None
        :type run_args: dict[str, str], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        :param container: container type for workload (e.g. "singularity"), defaults to None
        :type container: Container, optional
        """
        # Do not expand executable if running within a container
        if container:
            self.exe = [exe]
        else:
            self.exe = [expand_exe_path(exe)]

        self.exe_args = self._set_exe_args(exe_args)
        self.run_args = init_default({}, run_args, dict)
        self.env_vars = init_default({}, env_vars, dict)
        self.container = container
        self._run_command = run_command
        self.in_batch = False
        self.colocated_db_settings = None

    # To be overwritten by subclasses. Set of reserved args a user cannot change
    reserved_run_args = set()  # type: set[str]

    def set_nodes(self, nodes):
        """Set the number of nodes

        :param nodes: number of nodes to run with
        :type nodes: int
        """
        logger.warning(
            (
                "Node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_tasks(self, tasks):
        """Set the number of tasks to launch

        :param tasks: number of tasks to launch
        :type tasks: int
        """
        logger.warning(
            (
                "Task specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_tasks_per_node(self, tasks_per_node):
        """Set the number of tasks per node

        :param tasks_per_node: number of tasks to launch per node
        :type tasks_per_node: int
        """
        logger.warning(
            (
                "Task per node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_task_map(self, task_mapping):
        """Set a task mapping

        :param task_mapping: task mapping
        :type task_mapping: str
        """
        logger.warning(
            (
                "Task mapping specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_cpus_per_task(self, cpus_per_task):
        """Set the number of cpus per task

        :param cpus_per_task: number of cpus per task
        :type cpus_per_task: int
        """
        logger.warning(
            (
                "CPU per node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_hostlist(self, host_list):
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :type host_list: str | list[str]
        """
        logger.warning(
            (
                "Hostlist specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_hostlist_from_file(self, file_path):
        """Use the contents of a file to specify the hostlist for this job

        :param file_path: Path to the hostlist file
        :type file_path: str
        """
        logger.warning(
            (
                "Hostlist from file specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_excluded_hosts(self, host_list):
        """Specify a list of hosts to exclude for launching this job

        :param host_list: hosts to exclude
        :type host_list: str | list[str]
        """
        logger.warning(
            (
                "Excluded host list specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_cpu_bindings(self, bindings):
        """Set the cores to which MPI processes are bound

        :param bindings: List specifing the cores to which MPI processes are bound
        :type bindings: list[int] | int
        """
        logger.warning(
            (
                "CPU binding specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_memory_per_node(self, memory_per_node):
        """Set the amount of memory required per node in megabytes

        :param memory_per_node: Number of megabytes per node
        :type memory_per_node: int
        """
        logger.warning(
            (
                "Memory per node specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_verbose_launch(self, verbose):
        """Set the job to run in verbose mode

        :param verbose: Whether the job should be run verbosely
        :type verbose: bool
        """
        logger.warning(
            (
                "Verbose specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_quiet_launch(self, quiet):
        """Set the job to run in quiet mode

        :param quiet: Whether the job should be run quietly
        :type quiet: bool
        """
        logger.warning(
            (
                "Quiet specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_broadcast(self, dest_path=None):
        """Copy executable file to allocated compute nodes

        :param dest_path: Path to copy an executable file
        :type dest_path: str | None
        """
        logger.warning(
            (
                "Broadcast specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_time(self, hours=0, minutes=0, seconds=0):
        """Automatically format and set wall time

        :param hours: number of hours to run job
        :type hours: int
        :param minutes: number of minutes to run job
        :type minutes: int
        :param seconds: number of seconds to run job
        :type seconds: int
        """
        return self.set_walltime(
            self._fmt_walltime(int(hours), int(minutes), int(seconds))
        )

    def _fmt_walltime(self, hours, minutes, seconds):
        """Convert hours, minutes, and seconds into valid walltime format

        By defualt the formatted wall time is the total number of seconds.

        :param hours: number of hours to run job
        :type hours: int
        :param minutes: number of minutes to run job
        :type minutes: int
        :param seconds: number of seconds to run job
        :type seconds: int
        :returns: Formatted walltime
        :rtype: str
        """
        time_ = hours * 3600
        time_ += minutes * 60
        time_ += seconds
        return str(time_)

    def set_walltime(self, walltime):
        """Set the formatted walltime

        :param walltime: Time in format required by launcher``
        :type walltime: str
        """
        logger.warning(
            (
                "Walltime specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_binding(self, binding):
        """Set binding

        :param binding: Binding
        :type binding: str
        """
        logger.warning(
            (
                "binding specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def set_mpmd_preamble(self, preamble_lines):
        """Set preamble to a file to make a job MPMD

        :param preamble_lines: lines to put at the beginning of a file.
        :type preamble_lines: list[str]
        """
        logger.warning(
            (
                "MPMD preamble specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    def make_mpmd(self, settings):
        """Make job an MPMD job

        :param settings: ``RunSettings`` instance
        :type aprun_settings: RunSettings
        """
        logger.warning(
            (
                "Make MPMD specification not implemented for this "
                f"RunSettings type: {type(self)}"
            )
        )

    @property
    def run_command(self):
        """Return the launch binary used to launch the executable

        Attempt to expand the path to the executable if possible

        :returns: launch binary e.g. mpiexec
        :type: str | None
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

    def update_env(self, env_vars):
        """Update the job environment variables

        To fully inherit the current user environment, add the
        workload-manager-specific flag to the launch command through the
        :meth:`add_exe_args` method. For example, ``--export=ALL`` for
        slurm, or ``-V`` for PBS/aprun.


        :param env_vars: environment variables to update or add
        :type env_vars: dict[str, Union[str, int, float, bool]]
        :raises TypeError: if env_vars values cannot be coerced to strings
        """
        val_types = (str, int, float, bool)
        # Coerce env_vars values to str as a convenience to user
        for env, val in env_vars.items():
            if not isinstance(val, val_types):
                raise TypeError(
                    f"env_vars[{env}] was of type {type(val)}, not {val_types}"
                )
            else:
                self.env_vars[env] = val

    def add_exe_args(self, args):
        """Add executable arguments to executable

        :param args: executable arguments
        :type args: str | list[str]
        :raises TypeError: if exe args are not strings
        """
        if isinstance(args, str):
            args = args.split()
        for arg in args:
            if not isinstance(arg, str):
                raise TypeError("Executable arguments should be a list of str")
            self.exe_args.append(arg)

    def set(self, arg, value=None, condition=True):
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
            # returns ["exclusive", "None", "partition", "debug"] iff socket.gethostname()=="testing-system"
            # otherwise returns ["exclusive", "None"]

        :param arg: name of the argument
        :type arg: str
        :param value: value of the argument
        :type value: str | None
        :param conditon: set the argument if condition evaluates to True
        :type condition: bool
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

    def _set_exe_args(self, exe_args):
        if exe_args:
            if isinstance(exe_args, str):
                return exe_args.split()
            if isinstance(exe_args, list):
                plain_type = all([isinstance(arg, (str)) for arg in exe_args])
                if not plain_type:
                    nested_type = all(
                        [
                            all([isinstance(arg, (str)) for arg in exe_args_list])
                            for exe_args_list in exe_args
                        ]
                    )
                    if not nested_type:
                        raise TypeError(
                            "Executable arguments were not list of str or str"
                        )
                    else:
                        return exe_args
                return exe_args
            raise TypeError("Executable arguments were not list of str or str")
        else:
            return []

    def format_run_args(self):
        """Return formatted run arguments

        For ``RunSettings``, the run arguments are passed
        literally with no formatting.

        :return: list run arguments for these settings
        :rtype: list[str]
        """
        formatted = []
        for arg, value in self.run_args.items():
            formatted.append(arg)
            formatted.append(str(value))
        return formatted

    def format_env_vars(self):
        """Build environment variable string

        :returns: formatted list of strings to export variables
        :rtype: list[str]
        """
        formatted = []
        for key, val in self.env_vars.items():
            if val is None:
                formatted.append(f"{key}=")
            else:
                formatted.append(f"{key}={val}")
        return formatted

    def __str__(self):  # pragma: no-cover
        string = f"Executable: {self.exe[0]}\n"
        string += f"Executable Arguments: {' '.join((self.exe_args))}"
        if self.run_command:
            string += f"\nRun Command: {self._run_command}"
        if self.run_args:
            string += f"\nRun Arguments:\n{fmt_dict(self.run_args)}"
        if self.colocated_db_settings:
            string += "\nCo-located Database: True"
        return string


class BatchSettings:
    def __init__(self, batch_cmd, batch_args=None, **kwargs):
        self._batch_cmd = batch_cmd
        self.batch_args = init_default({}, batch_args, dict)
        self._preamble = []
        self.set_nodes(kwargs.get("nodes", None))
        self.set_walltime(kwargs.get("time", None))
        self.set_queue(kwargs.get("queue", None))
        self.set_account(kwargs.get("account", None))

    @property
    def batch_cmd(self):
        """Return the batch command

        Tests to see if we can expand the batch command
        path. If we can, then returns the expanded batch
        command. If we cannot, returns the batch command as is.

        :returns: batch command
        :type: str
        """
        if is_valid_cmd(self._batch_cmd):
            return expand_exe_path(self._batch_cmd)
        else:
            return self._batch_cmd

    def set_nodes(self, num_nodes):
        raise NotImplementedError

    def set_hostlist(self, host_list):
        raise NotImplementedError

    def set_queue(self, queue):
        raise NotImplementedError

    def set_walltime(self, walltime):
        raise NotImplementedError

    def set_account(self, account):
        raise NotImplementedError

    def format_batch_args(self):
        raise NotImplementedError

    def set_batch_command(self, command):
        """Set the command used to launch the batch e.g. ``sbatch``

        :param command: batch command
        :type command: str
        """
        self._batch_cmd = command

    def add_preamble(self, lines):
        """Add lines to the batch file preamble. The lines are just
        written (unmodified) at the beginning of the batch file
        (after the WLM directives) and can be used to e.g.
        start virtual environments before running the executables.

        :param line: lines to add to preamble.
        :type line: str or list[str]
        """
        if isinstance(lines, str):
            self._preamble += [lines]
        elif isinstance(lines, list):
            self._preamble += lines
        else:
            raise TypeError("Expected str or List[str] for lines argument")

    def __str__(self):  # pragma: no-cover
        string = f"Batch Command: {self._batch_cmd}"
        if self.batch_args:
            string += f"\nBatch arguments:\n{fmt_dict(self.batch_args)}"
        return string
