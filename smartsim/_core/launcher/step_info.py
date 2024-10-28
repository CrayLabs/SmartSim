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

import typing as t

import psutil

from ...status import JobStatus


class StepInfo:
    def __init__(
        self,
        status: JobStatus,
        launcher_status: str = "",
        returncode: t.Optional[int] = None,
        output: t.Optional[str] = None,
        error: t.Optional[str] = None,
    ) -> None:
        self.status = status
        self.launcher_status = launcher_status
        self.returncode = returncode
        self.output = output
        self.error = error

    def __str__(self) -> str:
        info_str = f"Status: {self.status.value}"
        info_str += f" | Launcher Status {self.launcher_status}"
        info_str += f" | Returncode {str(self.returncode)}"
        return info_str

    @property
    def mapping(self) -> t.Dict[str, JobStatus]:
        raise NotImplementedError

    def _get_smartsim_status(
        self, status: str, returncode: t.Optional[int] = None
    ) -> JobStatus:
        """
        Map the status of the WLM step to a smartsim-specific status
        """
        if any(ss_status.value == status for ss_status in JobStatus):
            return JobStatus(status)

        if status in self.mapping and returncode in [None, 0]:
            return self.mapping[status]

        return JobStatus.FAILED


class UnmanagedStepInfo(StepInfo):
    @property
    def mapping(self) -> t.Dict[str, JobStatus]:
        # see https://github.com/giampaolo/psutil/blob/master/psutil/_pslinux.py
        # see https://github.com/giampaolo/psutil/blob/master/psutil/_common.py
        return {
            psutil.STATUS_RUNNING: JobStatus.RUNNING,
            psutil.STATUS_SLEEPING: JobStatus.RUNNING,  # sleeping thread is still alive
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
        }

    def __init__(
        self,
        status: str = "",
        returncode: t.Optional[int] = None,
        output: t.Optional[str] = None,
        error: t.Optional[str] = None,
    ) -> None:
        smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )


class SlurmStepInfo(StepInfo):  # cov-slurm
    # see https://slurm.schedmd.com/squeue.html#lbAG
    mapping = {
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

    def __init__(
        self,
        status: str = "",
        returncode: t.Optional[int] = None,
        output: t.Optional[str] = None,
        error: t.Optional[str] = None,
    ) -> None:
        smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )


class PBSStepInfo(StepInfo):  # cov-pbs
    @property
    def mapping(self) -> t.Dict[str, JobStatus]:
        # pylint: disable-next=line-too-long
        # see http://nusc.nsu.ru/wiki/lib/exe/fetch.php/doc/pbs/PBSReferenceGuide19.2.1.pdf#M11.9.90788.PBSHeading1.81.Job.States
        return {
            "R": JobStatus.RUNNING,
            "B": JobStatus.RUNNING,
            "H": JobStatus.PAUSED,
            "M": (
                JobStatus.PAUSED
            ),  # Actually means that it was moved to another server,
            # TODO: understand what this implies
            "Q": JobStatus.PAUSED,
            "S": JobStatus.PAUSED,
            "T": JobStatus.PAUSED,  # This means in transition, see above for comment
            "U": JobStatus.PAUSED,
            "W": JobStatus.PAUSED,
            "E": JobStatus.COMPLETED,
            "F": JobStatus.COMPLETED,
            "X": JobStatus.COMPLETED,
        }

    def __init__(
        self,
        status: str = "",
        returncode: t.Optional[int] = None,
        output: t.Optional[str] = None,
        error: t.Optional[str] = None,
    ) -> None:
        if status == "NOTFOUND":
            if returncode is not None:
                smartsim_status = (
                    JobStatus.COMPLETED if returncode == 0 else JobStatus.FAILED
                )
            else:
                # if PBS job history isnt available, and job isnt in queue
                smartsim_status = JobStatus.COMPLETED
                returncode = 0
        else:
            smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )


class LSFBatchStepInfo(StepInfo):  # cov-lsf
    @property
    def mapping(self) -> t.Dict[str, JobStatus]:
        # pylint: disable-next=line-too-long
        # see https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=execution-about-job-states
        return {
            "RUN": JobStatus.RUNNING,
            "PSUSP": JobStatus.PAUSED,
            "USUSP": JobStatus.PAUSED,
            "SSUSP": JobStatus.PAUSED,
            "PEND": JobStatus.PAUSED,
            "DONE": JobStatus.COMPLETED,
        }

    def __init__(
        self,
        status: str = "",
        returncode: t.Optional[int] = None,
        output: t.Optional[str] = None,
        error: t.Optional[str] = None,
    ) -> None:
        if status == "NOTFOUND":
            if returncode is not None:
                smartsim_status = (
                    JobStatus.COMPLETED if returncode == 0 else JobStatus.FAILED
                )
            else:
                smartsim_status = JobStatus.COMPLETED
                returncode = 0
        else:
            smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )


class LSFJsrunStepInfo(StepInfo):  # cov-lsf
    @property
    def mapping(self) -> t.Dict[str, JobStatus]:
        # pylint: disable-next=line-too-long
        # see https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=execution-about-job-states
        return {
            "Killed": JobStatus.COMPLETED,
            "Running": JobStatus.RUNNING,
            "Queued": JobStatus.PAUSED,
            "Complete": JobStatus.COMPLETED,
        }

    def __init__(
        self,
        status: str = "",
        returncode: t.Optional[int] = None,
        output: t.Optional[str] = None,
        error: t.Optional[str] = None,
    ) -> None:
        if status == "NOTFOUND":
            if returncode is not None:
                smartsim_status = (
                    JobStatus.COMPLETED if returncode == 0 else JobStatus.FAILED
                )
            else:
                smartsim_status = JobStatus.COMPLETED
                returncode = 0
        else:
            smartsim_status = self._get_smartsim_status(status, returncode)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )


class SGEStepInfo(StepInfo):  # cov-pbs
    @property
    def mapping(self) -> t.Dict[str, JobStatus]:
        # pylint: disable-next=line-too-long
        # see https://manpages.ubuntu.com/manpages/jammy/man5/sge_status.5.html
        return {
            # Running states
            "r": JobStatus.RUNNING,
            "hr": JobStatus.RUNNING,
            "t": JobStatus.RUNNING,
            "Rr": JobStatus.RUNNING,
            "Rt": JobStatus.RUNNING,
            # Queued states
            "qw": JobStatus.QUEUED,
            "Rq": JobStatus.QUEUED,
            "hqw": JobStatus.QUEUED,
            "hRwq": JobStatus.QUEUED,
            # Paused states
            "s": JobStatus.PAUSED,
            "ts": JobStatus.PAUSED,
            "S": JobStatus.PAUSED,
            "tS": JobStatus.PAUSED,
            "T": JobStatus.PAUSED,
            "tT": JobStatus.PAUSED,
            "Rs": JobStatus.PAUSED,
            "Rts": JobStatus.PAUSED,
            "RS": JobStatus.PAUSED,
            "RtS": JobStatus.PAUSED,
            "RT": JobStatus.PAUSED,
            "RtT": JobStatus.PAUSED,
            # Failed states
            "Eqw": JobStatus.FAILED,
            "Ehqw": JobStatus.FAILED,
            "EhRqw": JobStatus.FAILED,
            # Finished states
            "z": JobStatus.COMPLETED,
            # Cancelled
            "dr": JobStatus.CANCELLED,
            "dt": JobStatus.CANCELLED,
            "dRr": JobStatus.CANCELLED,
            "dRt": JobStatus.CANCELLED,
            "ds": JobStatus.CANCELLED,
            "dS": JobStatus.CANCELLED,
            "dT": JobStatus.CANCELLED,
            "dRs": JobStatus.CANCELLED,
            "dRS": JobStatus.CANCELLED,
            "dRT": JobStatus.CANCELLED,
        }

    def __init__(
        self,
        status: str = "",
        returncode: t.Optional[int] = None,
        output: t.Optional[str] = None,
        error: t.Optional[str] = None,
    ) -> None:
        if status == "NOTFOUND":
            if returncode is not None:
                smartsim_status = (
                    JobStatus.COMPLETED if returncode == 0 else JobStatus.FAILED
                )
            else:
                # if PBS job history is not available, and job is not in queue
                smartsim_status = JobStatus.COMPLETED
                returncode = 0
        else:
            smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )
