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

import collections
import os
import os.path
import subprocess as sp
import time
import typing as t

from smartsim._core.config import CONFIG
from smartsim._core.launcher_.slurm import slurm_commands as commands
from smartsim._core.launcher_.slurm import slurm_parser as parser
from smartsim._core.utils import helpers, launcher
from smartsim.error import errors
from smartsim.log import get_logger
from smartsim.status import JobStatus
from smartsim.types import LaunchedJobID

if t.TYPE_CHECKING:
    from smartsim.experiment import Experiment


logger = get_logger(__name__)


class SrunCommand:
    """A type to inject job important information such as the job name and ID
    into an `srun` command so that it may be utilized to track the status of
    the job.
    """

    def __init__(
        self,
        name: str,
        srun_args: t.Sequence[str],
        executable: t.Sequence[str],
        job_id: str | None = None,
        environment: t.Mapping[str, str] | None = None,
    ) -> None:
        """Initialize a new trackable srun command.

        :param name: The name of the job.
        :param srun_args: Any command line args to feed to `srun`.
        :param executable: The command and any command line arguments that
            should be executed by `srun`.
        :param job_id: The id of the allocated job under which step should be
            executed.
        :param environment: Any additional environment variables to place in
            the environment before starting the `srun` subprocess.
        :raises errors.AllocationError: If the `job_id` was not supplied and
            could not be inferred.
        """
        self.name: t.Final = f"{name}-{helpers.create_short_id_str()}"
        self.srun_args: t.Final = tuple(srun_args)
        if job_id is None:
            try:
                job_id = os.environ["SLURM_JOB_ID"]
            except KeyError as e:
                raise errors.AllocationError(
                    "No allocation specified and could be found"
                ) from e
            logger.debug(f"Using allocation {job_id} gleaned from user environment")
        self.job_id: t.Final = job_id
        self.executable: t.Final = tuple(executable)
        self.env: t.Final[t.Mapping[str, str]] = (
            dict(environment) if environment is not None else {}
        )

    def as_command_line_args(self) -> tuple[str, ...]:
        """Format the `srun` command with job id and name information

        :returns: A sequence of symbols that can be opened in a subshell
        """
        srun = helpers.expand_exe_path("srun")
        return (
            srun,
            *self.srun_args,
            f"--job-name={self.name}",
            f"--jobid={self.job_id}",
            "--",
            *self.executable,
        )

    def start(self) -> None:
        """Start the `srun` command in a subshell"""
        # pylint: disable-next=consider-using-with
        sp.Popen(
            self.as_command_line_args(),
            env={**os.environ, **self.env},
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )


_SlurmCommandType: t.TypeAlias = SrunCommand
"""Types that are capable of being launched by the `SlurmLauncher`"""


class SlurmLauncher:
    """A launcher for launching/tracking slurm specific commands"""

    def __init__(
        self, *, launched: t.Mapping[LaunchedJobID, _LaunchedJobInfo] | None = None
    ) -> None:
        """Initialize a new slurm launcher.

        :param launched: Any previously launched slurm jobs that the launcher
            should be aware of. Primarily used for testing.
        """
        self._launched: t.Final = dict(launched) if launched is not None else {}

    @classmethod
    def create(cls, _: Experiment) -> SlurmLauncher:
        """Create a new launcher instance from an experiment instance.

        :param _: <Unused> An experiment instance.
        :returns: A new launcher instance.
        """
        return cls()

    def start(self, launchable: _SlurmCommandType, interval: int = 2) -> LaunchedJobID:
        """Have the slurm launcher start and track the progress of a new
        subprocess.

        :param launchable: The template of a slurm subprocess to start.
        :param interval: The amount of time in seconds to wait between `sacct`
            calls to get the step id.
        :returns: An id to reference the process for status tracking.
        """
        launchable.start()
        trials = CONFIG.wlm_trials
        step_id = None
        while step_id is None and trials > 0:
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # TODO: Really don't like the implied sacct format and required
            #       order to call these fns in order to get the step id
            # -------------------------------------------------------------------------
            out, _ = commands.sacct(
                ["--noheader", "-p", "--format=jobname,jobid"], raise_on_err=True
            )
            step_id = parser.parse_step_id_from_sacct(out, launchable.name)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            trials -= 1
            if step_id is None:
                time.sleep(interval)
        if step_id is None:
            raise errors.LauncherError("Could not find id of launched job step")
        id_ = launcher.create_job_id()
        self._launched[id_] = _LaunchedJobInfo(step_id, launchable.name)
        return id_

    def _get_slurm_info_from_job_id(self, id_: LaunchedJobID, /) -> _LaunchedJobInfo:
        """Find the info for a slurm subprocess given a launched job id issued
        to a user.

        :param id_: The job id issued to the user.
        :raises errors.LauncherJobNotFound: The id is not recognized.
        :returns: Info about the launched slurm job.
        """
        if (info := self._launched.get(id_)) is None:
            msg = f"Launcher `{self}` has not launched a job with id `{id_}`"
            raise errors.LauncherJobNotFound(msg)
        return info

    def get_status(self, *ids: LaunchedJobID) -> t.Mapping[LaunchedJobID, JobStatus]:
        """Take a collection of job ids and return the status of the
        corresponding slrum processes started by the slurm launcher.

        :param ids: A collection of ids of the launched jobs to get the
            statuses of.
        :returns: A mapping of ids for jobs to stop to their reported status.
        """
        id_to_info = {id_: self._get_slurm_info_from_job_id(id_) for id_ in ids}

        def status_override(info: _LaunchedJobInfo) -> JobStatus | None:
            return info.status_override

        status_to_infos = helpers.group_by(status_override, id_to_info.values())
        needs_fetch = status_to_infos.get(None, ())
        fetched_status = (
            self._get_status(*(info.slurm_id for info in needs_fetch))
            if needs_fetch
            else {}
        )
        has_overwrite = {
            info.slurm_id: status
            for status, infos in status_to_infos.items()
            for info in infos
            if status is not None
        }
        slurm_ids_to_status = collections.ChainMap(has_overwrite, fetched_status)
        return {id_: slurm_ids_to_status[id_to_info[id_].slurm_id] for id_ in ids}

    @staticmethod
    def _get_status(
        id_: parser.StepID, *ids: parser.StepID
    ) -> dict[parser.StepID, JobStatus]:
        """Given a collection of step ids, interogate slurm for the status of
        the steps.

        :param id_: The first step id to get the status for.
        :param ids: Any additional step ids to get the status for.
        :returns: A mapping of step ids to statuses
        """
        ids = (id_,) + ids
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # TODO: Really don't like the implied sacct format and required
        #       order to call these fns in order to get the status
        # -------------------------------------------------------------------------
        out, _ = commands.sacct(
            ["--noheader", "-p", "-b", "--jobs", ",".join(ids)], raise_on_err=True
        )
        stats = ((id_, parser.parse_sacct(out, id_)[0]) for id_ in ids)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        to_ss_stat = {
            "RUNNING": JobStatus.RUNNING,
            "CONFIGURING": JobStatus.RUNNING,
            "STAGE_OUT": JobStatus.RUNNING,
            "COMPLETED": JobStatus.COMPLETED,
            "DEADLINE": JobStatus.COMPLETED,
            "TIMEOUT": JobStatus.COMPLETED,
            "BOOT_FAIL": JobStatus.FAILED,
            "FAILED": JobStatus.FAILED,
            "NODE_FAIL": JobStatus.FAILED,
            "OUT_OF_MEMORY": JobStatus.FAILED,
            "CANCELLED": JobStatus.CANCELLED,
            "CANCELLED+": JobStatus.CANCELLED,
            "REVOKED": JobStatus.CANCELLED,
            "PENDING": JobStatus.PAUSED,
            "PREEMPTED": JobStatus.PAUSED,
            "RESV_DEL_HOLD": JobStatus.PAUSED,
            "REQUEUE_FED": JobStatus.PAUSED,
            "REQUEUE_HOLD": JobStatus.PAUSED,
            "REQUEUED": JobStatus.PAUSED,
            "RESIZING": JobStatus.PAUSED,
            "SIGNALING": JobStatus.PAUSED,
            "SPECIAL_EXIT": JobStatus.PAUSED,
            "STOPPED": JobStatus.PAUSED,
            "SUSPENDED": JobStatus.PAUSED,
        }
        return {id_: to_ss_stat.get(stat, JobStatus.UNKNOWN) for id_, stat in stats}

    def stop_jobs(self, *ids: LaunchedJobID) -> t.Mapping[LaunchedJobID, JobStatus]:
        """Take a collection of job ids and kill the corresponding processes
        started by the slurm launcher.

        :param ids: The ids of the launched jobs to stop.
        :returns: A mapping of ids to their reported status after attempting to
            stop them.
        """
        slurm_infos = tuple(map(self._get_slurm_info_from_job_id, ids))
        for info in slurm_infos:
            self._stop_job(info)
        return self.get_status(*ids)

    @staticmethod
    def _stop_job(job_info: _LaunchedJobInfo) -> None:
        """Given the launch information for a slurm process, attempt to kill
        that process.

        :param job_info: The info for the job that the launcher should attempt
            to kill.
        """
        step_id = job_info.slurm_id
        # Check if step_id is part of colon-separated run, this is reflected in
        # a '+' in the step id, so that the format becomes 12345+1.0.  If we
        # find it it can mean two things: a MPMD srun command, or a
        # heterogeneous job.  If it is a MPMD srun, then stop parent step
        # because sub-steps cannot be stopped singularly.
        is_sub_step = "+" in step_id
        is_het_job = os.getenv("SLURM_HET_SIZE") is not None
        # If it is a heterogeneous job, we can stop them like this. Slurm will
        # throw an error, but will actually kill steps correctly.
        if is_sub_step and not is_het_job:
            step_id_, *_ = step_id.split("+", maxsplit=1)
            step_id = parser.StepID(step_id_)  # Ugly cast for type check
        ret_code, _, err = commands.scancel([step_id])
        if ret_code != 0:
            if is_het_job:
                msg = (
                    "SmartSim received a non-zero exit code while canceling"
                    f" a heterogeneous job step {job_info.name}!\n"
                    "The following error might be internal to Slurm\n"
                    "and the heterogeneous job step could have been correctly"
                    " canceled.\n"
                    "SmartSim will consider it canceled.\n"
                )
                job_info.status_override = JobStatus.CANCELLED
            else:
                msg = f"Unable to cancel job step {job_info.name}\n{err}"
            logger.warning(msg)


class _LaunchedJobInfo:
    """Slurm specific launch information for a launched job"""

    def __init__(
        self,
        slurm_id: parser.StepID,
        name: str,
        *,
        status_override: JobStatus | None = None,
    ) -> None:
        self.slurm_id: t.Final = slurm_id
        self.name: t.Final = name
        self.status_override = status_override
