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

import io
import pathlib
import subprocess as sp
import typing as t

import psutil

from smartsim._core.arguments.shell import ShellLaunchArguments
from smartsim._core.dispatch import EnvironMappingType, FormatterType, WorkingDirectory
from smartsim._core.utils import helpers
from smartsim._core.utils.launcher import create_job_id
from smartsim.error import errors
from smartsim.log import get_logger
from smartsim.settings.arguments.launch_arguments import LaunchArguments
from smartsim.status import JobStatus
from smartsim.types import LaunchedJobID

if t.TYPE_CHECKING:
    from typing_extensions import Self

    from smartsim.experiment import Experiment

logger = get_logger(__name__)


class ShellLauncherCommand(t.NamedTuple):
    env: EnvironMappingType
    path: pathlib.Path
    stdout: io.TextIOWrapper | int
    stderr: io.TextIOWrapper | int
    command_tuple: t.Sequence[str]


def make_shell_format_fn(
    run_command: str | None,
) -> FormatterType[ShellLaunchArguments, ShellLauncherCommand]:
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
        args: ShellLaunchArguments,
        exe: t.Sequence[str],
        path: WorkingDirectory,
        env: EnvironMappingType,
        stdout_path: pathlib.Path,
        stderr_path: pathlib.Path,
    ) -> ShellLauncherCommand:
        command_tuple = (
            (
                run_command,
                *(args.format_launch_args() or ()),
                "--",
                *exe,
            )
            if run_command is not None
            else exe
        )
        # pylint: disable-next=consider-using-with
        return ShellLauncherCommand(
            env, pathlib.Path(path), open(stdout_path), open(stderr_path), command_tuple
        )

    return impl


class ShellLauncher:
    """A launcher for launching/tracking local shell commands"""

    def __init__(self) -> None:
        """Initialize a new shell launcher."""
        self._launched: dict[LaunchedJobID, sp.Popen[bytes]] = {}

    def check_popen_inputs(self, shell_command: ShellLauncherCommand) -> None:
        """Validate that the contents of a shell command are valid.

        :param shell_command: The command to validate
        :raises ValueError: If the command is not valid
        """
        if not shell_command.path.exists():
            raise ValueError("Please provide a valid path to ShellLauncherCommand.")

    def start(self, shell_command: ShellLauncherCommand) -> LaunchedJobID:
        """Have the shell launcher start and track the progress of a new
        subprocess.

        :param shell_command: The template of a subprocess to start.
        :returns: An id to reference the process for status.
        """
        self.check_popen_inputs(shell_command)
        id_ = create_job_id()
        exe, *rest = shell_command.command_tuple
        expanded_exe = helpers.expand_exe_path(exe)
        # pylint: disable-next=consider-using-with
        self._launched[id_] = sp.Popen(
            (expanded_exe, *rest),
            cwd=shell_command.path,
            env={k: v for k, v in shell_command.env.items() if v is not None},
            stdout=shell_command.stdout,
            stderr=shell_command.stderr,
        )
        return id_

    def _get_proc_from_job_id(self, id_: LaunchedJobID, /) -> sp.Popen[bytes]:
        """Given an issued job id, return the process represented by that id.

        :param id_: The launched job id of the process
        :raises: errors.LauncherJobNotFound: The id could not be mapped to a
            process. This usually means that the provided id was not issued by
            this launcher instance.
        :returns: The process that the shell launcher started and represented
            by the issued id.
        """
        if (proc := self._launched.get(id_)) is None:
            msg = f"Launcher `{self}` has not launched a job with id `{id_}`"
            raise errors.LauncherJobNotFound(msg)
        return proc

    def get_status(
        self, *launched_ids: LaunchedJobID
    ) -> t.Mapping[LaunchedJobID, JobStatus]:
        """Take a collection of job ids and return the status of the
        corresponding processes started by the shell launcher.

        :param launched_ids: A collection of ids of the launched jobs to get
            the statuses of.
        :returns: A mapping of ids for jobs to stop to their reported status.
        """
        return {id_: self._get_status(id_) for id_ in launched_ids}

    def _get_status(self, id_: LaunchedJobID, /) -> JobStatus:
        """Given an issued job id, return the process represented by that id

        :param id_: The launched job id of the process to get the status of.
        :returns: The status of that process represented by the given id.
        """
        proc = self._get_proc_from_job_id(id_)
        ret_code = proc.poll()
        if ret_code is None:
            status = psutil.Process(proc.pid).status()
            return {
                psutil.STATUS_RUNNING: JobStatus.RUNNING,
                psutil.STATUS_SLEEPING: JobStatus.RUNNING,
                psutil.STATUS_WAKING: JobStatus.RUNNING,
                psutil.STATUS_DISK_SLEEP: JobStatus.RUNNING,
                psutil.STATUS_DEAD: JobStatus.FAILED,
                psutil.STATUS_TRACING_STOP: JobStatus.PAUSED,
                psutil.STATUS_WAITING: JobStatus.PAUSED,
                psutil.STATUS_STOPPED: JobStatus.PAUSED,
                psutil.STATUS_LOCKED: JobStatus.PAUSED,
                psutil.STATUS_PARKED: JobStatus.PAUSED,
                psutil.STATUS_IDLE: JobStatus.PAUSED,
                psutil.STATUS_ZOMBIE: JobStatus.COMPLETED,
            }.get(status, JobStatus.UNKNOWN)
        if ret_code == 0:
            return JobStatus.COMPLETED
        return JobStatus.FAILED

    def stop_jobs(
        self, *launched_ids: LaunchedJobID
    ) -> t.Mapping[LaunchedJobID, JobStatus]:
        """Take a collection of job ids and kill the corresponding processes
        started by the shell launcher.

        :param launched_ids: The ids of the launched jobs to stop.
        :returns: A mapping of ids for jobs to stop to their reported status
            after attempting to stop them.
        """
        return {id_: self._stop(id_) for id_ in launched_ids}

    def _stop(self, id_: LaunchedJobID, /, wait_time: float = 5.0) -> JobStatus:
        """Stop a job represented by an id

        The launcher will first start by attempting to kill the process using
        by sending a SIGTERM signal and then waiting for an amount of time. If
        the process is not killed by the timeout time, a SIGKILL signal will be
        sent and another waiting period will be started. If the period also
        ends, the message will be logged and the process will be left to
        continue running. The method will then get and return the status of the
        job.

        :param id_: The id of a launched job to stop.
        :param wait: The maximum amount of time, in seconds, to wait for a
            signal to stop a process.
        :returns: The status of the job after sending signals to terminate the
            started process.
        """
        proc = self._get_proc_from_job_id(id_)
        if proc.poll() is None:
            msg = f"Attempting to terminate local process {proc.pid}"
            logger.debug(msg)
            proc.terminate()

        try:
            proc.wait(wait_time)
        except sp.TimeoutExpired:
            msg = f"Failed to terminate process {proc.pid}. Attempting to kill."
            logger.warning(msg)
            proc.kill()

        try:
            proc.wait(wait_time)
        except sp.TimeoutExpired:
            logger.error(f"Failed to kill process {proc.pid}")
        return self._get_status(id_)

    @classmethod
    def create(cls, _: Experiment) -> Self:
        """Create a new launcher instance from an experiment instance.

        :param _: <Unused> An experiment instance.
        :returns: A new launcher instance.
        """
        return cls()
