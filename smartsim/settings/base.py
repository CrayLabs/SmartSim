# BSD 2-Clause License
#
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
        :param reserved_run_args: set of arguments that should not be set by user, defaults to None
        :type run_args: set[str], optional
        :param env_vars: environment vars to launch job with, defaults to None
        :type env_vars: dict[str, str], optional
        """
        self.exe = [expand_exe_path(exe)]
        self.exe_args = self._set_exe_args(exe_args)
        self.run_args = init_default({}, run_args, dict)
        self.env_vars = init_default({}, env_vars, dict)
        self._run_command = run_command
        self.in_batch = False
        self.colocated_db_settings = None

    reserved_run_args = set()

    def set_tasks(self, tasks):
        """Set the number of tasks to launch

        :param tasks: number of tasks to launch
        :type tasks: int
        """
        raise NotImplementedError(
            f"Task specification not implemented for this RunSettings type: {type(self)}"
        )

    def set_tasks_per_node(self, tasks_per_node):
        """Set the number of tasks per node

        :param tasks_per_node: number of tasks to launch per node
        :type tasks_per_node: int
        """
        raise NotImplementedError(
            f"Task per node specification not implemented for this RunSettings type: {type(self)}"
        )

    def set_cpus_per_task(self, cpus_per_task):
        """Set the number of cpus per task

        :param cpus_per_task: number of cpus per task
        :type cpus_per_task: int
        """
        raise NotImplementedError(
            f"CPU per node specification not implemented for this RunSettings type: {type(self)}"
        )

    def set_hostlist(self, host_list):
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :type host_list: str | list[str]
        """
        raise NotImplementedError(
            f"Host list specification not implemented for this RunSettings type: {type(self)}"
        )

    @property
    def run_command(self):
        """Return the launch binary used to launch the executable

        Attempt to expand the path to the executable if possible

        :returns: launch binary e.g. mpiexec
        :type: str | None
        """
        if self._run_command:
            if is_valid_cmd(self._run_command):
                # command is valid and will be expanded
                return expand_exe_path(self._run_command)
            # command is not valid, so return it as is
            # it may be on the compute nodes but not local machine
            return self._run_command
        # run without run command
        return None

    def update_env(self, env_vars):
        """Update the job environment variables

        :param env_vars: environment variables to update or add
        :type env_vars: dict[str, str]
        """
        self.env_vars.update(env_vars)

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
        """allows users to set individual run arguments. Does
        basic formating such as stripping leading dashes.

        This method also ensures that the argument being set
        is a valid argument for the user to be manually adjusting.
        If it is not the argument is not set and a warning is logged.
        This is mainly used by subclasses to prevent users from changing
        settings for a specific launcher that would interfere with SmartSim.

        Basic Usage

        .. highlight:: python
        .. code-block:: python

            rs = RunSettings("python")
            rs.set("an-arg", "a-val")
            rs.set("a-flag")
            rs.run_args["an-arg"]  # returns "a-val"
            rs.run_args["a-flag"]  # returns "a-flag"

        Subclasses Restricting Arguments

        .. highlight:: python
        .. code-block:: python

            ReservedArgsSettings(RunSettings)
                reserved_args = {"no-set"}
            
            ras = ReservedArgsSettings("python")
            ras.set("no-set")  # logs a warining
            "no-set" in ras.run_args  # returns False

        Agument String Formatting

        .. highlight:: python
        .. code-block:: python

            rs.set("--run-arg")
            "run-arg" in rs.run_args  # returns True

        Conditional Setting

        .. highlight:: python
        .. code-block:: python

            launcher = "slurm"
            rs.set("conditional-arg", "val", launcher=="cobalt")
            "conditional-arg" in rs.run_args  # returns False

        :param arg: name of the argument
        :type arg: str
        :param value: value of the argument
        :type value: str | None
        :param conditon: optional argument to conditionally set the argument
        :type condition: bool
        """
        if not isinstance(arg, str):
            raise TypeError("Argument name should be of type str")
        if value is not None and not isinstance(value, str):
            raise TypeError("Argument value should be of type str or None")
        arg = arg.strip().lstrip("-")
        if arg in self.reserved_run_args:
            logger.warning(
                (
                    f"Could not set argument '{arg}': "
                    f"it is a reserved arguement of '{type(self).__name__}'"
                )
            )
            return
        if not condition:
            logger.info(f"Could not set argument '{arg}': condition not met")
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
