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
from ..utils import get_logger
from ..utils.helpers import expand_exe_path, init_default

logger = get_logger(__name__)


class RunSettings:
    def __init__(
        self, exe, exe_args=None, run_command="", run_args=None, env_vars=None
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
        :param exe_args: executable arguments
        :type exe_args: str | list[str], optional
        :param run_command: launch binary e.g. srun
        :type run_command: str, optional
        :param run_args: arguments for run command (e.g. `-np` for `mpiexec`)
        :type run_args: dict[str, str], optional
        :param env_vars: environment vars to launch job with
        :type env_vars: dict[str, str], optional
        """
        self.exe = [expand_exe_path(exe)]
        self.exe_args = self._set_exe_args(exe_args)
        self.run_args = init_default({}, run_args, (dict, list))
        self.env_vars = init_default({}, env_vars, (dict, list))
        self._run_command = run_command
        self.in_batch = False

    @property
    def run_command(self):
        """Return the launch binary used to launch the executable

        :returns: launch binary e.g. mpiexec
        :type: str
        """
        try:
            if self._run_command:
                cmd = expand_exe_path(self._run_command)
                return cmd
            return None
        except SSConfigError:
            return self._run_command

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

    def __str__(self):
        string = f"Executable: {self.exe[0]}\n"
        string += f"Executable arguments: {self.exe_args}\n"
        if self.run_command:
            string += f"Run Command: {self._run_command}\n"
        if self.run_args:
            string += f"Run arguments: {pformat(self.run_args)}"
        return string


class BatchSettings:
    def __init__(self, batch_cmd, batch_args=None):
        self._batch_cmd = batch_cmd
        self.batch_args = init_default({}, batch_args, dict)
        self._preamble = []

    @property
    def batch_cmd(self):
        """Return the batch command

        Tests to see if we can expand the batch command
        path, and if not, returns the batch command as is

        :returns: batch command
        :type: str
        """
        try:
            cmd = expand_exe_path(self._batch_cmd)
            return cmd
        except SSConfigError:
            return self._batch_cmd

    def set_nodes(self, num_nodes):
        raise NotImplementedError

    def set_walltime(self, walltime):
        raise NotImplementedError

    def set_account(self, acct):
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
            raise TypeError

    def __str__(self):
        string = f"Batch Command: {self._batch_cmd}\n"
        if self.batch_args:
            string += f"Batch arguments: {pformat(self.batch_args)}"
        return string
