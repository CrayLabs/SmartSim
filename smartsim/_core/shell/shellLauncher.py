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

import typing as t

from smartsim.settings.arguments.launchArguments import LaunchArguments
from smartsim.types import LaunchedJobID

from smartsim.log import get_logger

import subprocess as sp

from smartsim._core.utils import helpers
from smartsim._core.dispatch import create_job_id, ExecutableProtocol, dispatch, _FormatterType, _EnvironMappingType

if t.TYPE_CHECKING:
    from typing_extensions import Self
    from smartsim.experiment import Experiment

logger = get_logger(__name__)

class ShellLauncher:
    """Mock launcher for launching/tracking simple shell commands"""

    def __init__(self) -> None:
        self._launched: dict[LaunchedJobID, sp.Popen[bytes]] = {}

    def start(self, command: t.Sequence[str]) -> LaunchedJobID:
        id_ = create_job_id()
        exe, *rest = command
        # pylint: disable-next=consider-using-with
        self._launched[id_] = sp.Popen((helpers.expand_exe_path(exe), *rest))
        return id_

    @classmethod
    def create(cls, _: Experiment) -> Self:
        return cls()

    @staticmethod
    def make_shell_format_fn(
        run_command: str | None,
    ) -> _FormatterType[LaunchArguments, t.Sequence[str]]:
        """A function that builds a function that formats a `LaunchArguments` as a
        shell executable sequence of strings for a given launching utility.

        Example usage:

        .. highlight:: python
        .. code-block:: python

            echo_hello_world: ExecutableProtocol = ...
            env = {}
            slurm_args: SlurmLaunchArguments = ...
            slurm_args.set_nodes(3)

            as_srun_command = make_shell_format_fn("srun")
            fmt_cmd = as_srun_command(slurm_args, echo_hello_world, env)
            print(list(fmt_cmd))
            # prints: "['srun', '--nodes=3', '--', 'echo', 'Hello World!']"

        .. note::
            This function was/is a kind of slap-dash implementation, and is likely
            to change or be removed entierely as more functionality is added to the
            shell launcher. Use with caution and at your own risk!

        :param run_command: Name or path of the launching utility to invoke with
            the arguments.
        :returns: A function to format an arguments, an executable, and an
            environment as a shell launchable sequence for strings.
        """

        def impl(
            args: LaunchArguments, exe: ExecutableProtocol, _env: _EnvironMappingType
        ) -> t.Sequence[str]:
            return (
                (
                    run_command,
                    *(args.format_launch_args() or ()),
                    "--",
                    *exe.as_program_arguments(),
                )
                if run_command is not None
                else exe.as_program_arguments()
            )

        return impl
