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
        self, exe, exe_args=None, run_command="", run_args=None, env_vars=None, **kwargs
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
        """
        self.exe = [expand_exe_path(exe)]
        self.exe_args = self._set_exe_args(exe_args)
        self.run_args = init_default({}, run_args, (dict, list))
        self.env_vars = init_default({}, env_vars, (dict, list))
        self._run_command = run_command
        self.in_batch = False

    def set_tasks(self, tasks):
        """Set the number of tasks to launch

        :param tasks: number of tasks to launch
        :type tasks: int
        """
        raise NotImplementedError(
            f"Task specification not implemented for this RunSettings type: {type(self)}")

    def set_tasks_per_node(self, tasks_per_node):
        """Set the number of tasks per node

        :param tasks_per_node: number of tasks to launch per node
        :type tasks_per_node: int
        """
        raise NotImplementedError(
            f"Task per node specification not implemented for this RunSettings type: {type(self)}")

    def set_cpus_per_task(self, cpus_per_task):
        """Set the number of cpus per task

        :param cpus_per_task: number of cpus per task
        :type cpus_per_task: int
        """
        raise NotImplementedError(
            f"CPU per node specification not implemented for this RunSettings type: {type(self)}")

    def set_hostlist(self, host_list):
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :type host_list: str | list[str]
        """
        raise NotImplementedError(
            f"Host list specification not implemented for this RunSettings type: {type(self)}")

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
            run_command = _detect_command(launcher)

    # if user specified and supported or auto detection worked
    if run_command and run_command in supported:
        return supported[run_command](exe, exe_args, run_args, env_vars, **kwargs)

    # 1) user specified and not implementation in SmartSim
    # 2) user supplied run_command=None
    # 3) local launcher being used and default of "auto" was passed.
    return RunSettings(exe, exe_args, run_command, run_args, env_vars)
