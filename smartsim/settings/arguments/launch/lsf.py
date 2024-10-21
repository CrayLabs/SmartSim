# BSD 2-Clause License
#
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

import pathlib
import subprocess
import typing as t

from smartsim._core.arguments.shell import ShellLaunchArguments
from smartsim._core.dispatch import EnvironMappingType, dispatch
from smartsim._core.shell.shell_launcher import ShellLauncher, ShellLauncherCommand
from smartsim.log import get_logger

from ...common import set_check_input
from ...launch_command import LauncherType

logger = get_logger(__name__)


def _as_jsrun_command(
    args: ShellLaunchArguments,
    exe: t.Sequence[str],
    path: pathlib.Path,
    env: EnvironMappingType,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
) -> ShellLauncherCommand:
    command_tuple = (
        "jsrun",
        *(args.format_launch_args() or ()),
        f"--stdio_stdout={stdout_path}",
        f"--stdio_stderr={stderr_path}",
        "--",
        *exe,
    )
    return ShellLauncherCommand(
        env, path, subprocess.DEVNULL, subprocess.DEVNULL, command_tuple
    )


@dispatch(with_format=_as_jsrun_command, to_launcher=ShellLauncher)
class JsrunLaunchArguments(ShellLaunchArguments):
    def launcher_str(self) -> str:
        """Get the string representation of the launcher

        :returns: The string representation of the launcher
        """
        return LauncherType.Lsf.value

    def _reserved_launch_args(self) -> set[str]:
        """Return reserved launch arguments.

        :returns: The set of reserved launcher arguments
        """
        return {"chdir", "h", "stdio_stdout", "o", "stdio_stderr", "k"}

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        This sets ``--np``

        :param tasks: number of tasks
        """
        self.set("np", str(tasks))

    def set_binding(self, binding: str) -> None:
        """Set binding

        This sets ``--bind``

        :param binding: Binding, e.g. `packed:21`
        """
        self.set("bind", binding)

    def format_env_vars(self, env_vars: t.Mapping[str, str | None]) -> list[str]:
        """Format environment variables. Each variable needs
        to be passed with ``--env``. If a variable is set to ``None``,
        its value is propagated from the current environment.

        :returns: formatted list of strings to export variables
        """
        format_str = []
        for k, v in env_vars.items():
            if v:
                format_str += ["-E", f"{k}={v}"]
            else:
                format_str += ["-E", f"{k}"]
        return format_str

    def format_launch_args(self) -> t.List[str]:
        """Return a list of LSF formatted run arguments

        :return: list of LSF arguments for these settings
        """
        # args launcher uses
        args = []

        for opt, value in self._launch_args.items():
            short_arg = bool(len(str(opt)) == 1)
            prefix = "-" if short_arg else "--"
            if value is None:
                args += [prefix + opt]
            else:
                if short_arg:
                    args += [prefix + opt, str(value)]
                else:
                    args += ["=".join((prefix + opt, str(value)))]
        return args

    def set(self, key: str, value: str | None) -> None:
        """Set an arbitrary launch argument

        :param key: The launch argument
        :param value: A string representation of the value for the launch
            argument (if applicable), otherwise `None`
        """
        set_check_input(key, value)
        if key in self._reserved_launch_args():
            logger.warning(
                (
                    f"Could not set argument '{key}': "
                    f"it is a reserved argument of '{type(self).__name__}'"
                )
            )
            return
        if key in self._launch_args and key != self._launch_args[key]:
            logger.warning(f"Overwritting argument '{key}' with value '{value}'")
        self._launch_args[key] = value
