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

from ...status import SmartSimStatus


class StepInfo:
    def __init__(
        self,
        status: SmartSimStatus,
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
    def mapping(self) -> t.Dict[str, SmartSimStatus]:
        raise NotImplementedError

    def _get_smartsim_status(
        self, status: str, returncode: t.Optional[int] = None
    ) -> SmartSimStatus:
        """
        Map the status of the WLM step to a smartsim-specific status
        """
        if any(ss_status.value == status for ss_status in SmartSimStatus):
            return SmartSimStatus(status)

        if status in self.mapping and returncode in [None, 0]:
            return self.mapping[status]

        return SmartSimStatus.STATUS_FAILED


class UnmanagedStepInfo(StepInfo):
    @property
    def mapping(self) -> t.Dict[str, SmartSimStatus]:
        # see https://github.com/giampaolo/psutil/blob/master/psutil/_pslinux.py
        # see https://github.com/giampaolo/psutil/blob/master/psutil/_common.py
        return {
            psutil.STATUS_RUNNING: SmartSimStatus.STATUS_RUNNING,
            psutil.STATUS_SLEEPING: (
                SmartSimStatus.STATUS_RUNNING
            ),  # sleeping thread is still alive
            psutil.STATUS_WAKING: SmartSimStatus.STATUS_RUNNING,
            psutil.STATUS_DISK_SLEEP: SmartSimStatus.STATUS_RUNNING,
            psutil.STATUS_DEAD: SmartSimStatus.STATUS_FAILED,
            psutil.STATUS_TRACING_STOP: SmartSimStatus.STATUS_PAUSED,
            psutil.STATUS_WAITING: SmartSimStatus.STATUS_PAUSED,
            psutil.STATUS_STOPPED: SmartSimStatus.STATUS_PAUSED,
            psutil.STATUS_LOCKED: SmartSimStatus.STATUS_PAUSED,
            psutil.STATUS_PARKED: SmartSimStatus.STATUS_PAUSED,
            psutil.STATUS_IDLE: SmartSimStatus.STATUS_PAUSED,
            psutil.STATUS_ZOMBIE: SmartSimStatus.STATUS_COMPLETED,
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
        "RUNNING": SmartSimStatus.STATUS_RUNNING,
        "CONFIGURING": SmartSimStatus.STATUS_RUNNING,
        "STAGE_OUT": SmartSimStatus.STATUS_RUNNING,
        "COMPLETED": SmartSimStatus.STATUS_COMPLETED,
        "DEADLINE": SmartSimStatus.STATUS_COMPLETED,
        "TIMEOUT": SmartSimStatus.STATUS_COMPLETED,
        "BOOT_FAIL": SmartSimStatus.STATUS_FAILED,
        "FAILED": SmartSimStatus.STATUS_FAILED,
        "NODE_FAIL": SmartSimStatus.STATUS_FAILED,
        "OUT_OF_MEMORY": SmartSimStatus.STATUS_FAILED,
        "CANCELLED": SmartSimStatus.STATUS_CANCELLED,
        "CANCELLED+": SmartSimStatus.STATUS_CANCELLED,
        "REVOKED": SmartSimStatus.STATUS_CANCELLED,
        "PENDING": SmartSimStatus.STATUS_PAUSED,
        "PREEMPTED": SmartSimStatus.STATUS_PAUSED,
        "RESV_DEL_HOLD": SmartSimStatus.STATUS_PAUSED,
        "REQUEUE_FED": SmartSimStatus.STATUS_PAUSED,
        "REQUEUE_HOLD": SmartSimStatus.STATUS_PAUSED,
        "REQUEUED": SmartSimStatus.STATUS_PAUSED,
        "RESIZING": SmartSimStatus.STATUS_PAUSED,
        "SIGNALING": SmartSimStatus.STATUS_PAUSED,
        "SPECIAL_EXIT": SmartSimStatus.STATUS_PAUSED,
        "STOPPED": SmartSimStatus.STATUS_PAUSED,
        "SUSPENDED": SmartSimStatus.STATUS_PAUSED,
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
    def mapping(self) -> t.Dict[str, SmartSimStatus]:
        # pylint: disable=line-too-long
        # see http://nusc.nsu.ru/wiki/lib/exe/fetch.php/doc/pbs/PBSReferenceGuide19.2.1.pdf#M11.9.90788.PBSHeading1.81.Job.States
        return {
            "R": SmartSimStatus.STATUS_RUNNING,
            "B": SmartSimStatus.STATUS_RUNNING,
            "H": SmartSimStatus.STATUS_PAUSED,
            "M": (
                SmartSimStatus.STATUS_PAUSED
            ),  # Actually means that it was moved to another server,
            # TODO: understand what this implies
            "Q": SmartSimStatus.STATUS_PAUSED,
            "S": SmartSimStatus.STATUS_PAUSED,
            "T": (
                SmartSimStatus.STATUS_PAUSED
            ),  # This means in transition, see above for comment
            "U": SmartSimStatus.STATUS_PAUSED,
            "W": SmartSimStatus.STATUS_PAUSED,
            "E": SmartSimStatus.STATUS_COMPLETED,
            "F": SmartSimStatus.STATUS_COMPLETED,
            "X": SmartSimStatus.STATUS_COMPLETED,
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
                    SmartSimStatus.STATUS_COMPLETED
                    if returncode == 0
                    else SmartSimStatus.STATUS_FAILED
                )
            else:
                # if PBS job history isnt available, and job isnt in queue
                smartsim_status = SmartSimStatus.STATUS_COMPLETED
                returncode = 0
        else:
            smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )


class LSFBatchStepInfo(StepInfo):  # cov-lsf
    @property
    def mapping(self) -> t.Dict[str, SmartSimStatus]:
        # pylint: disable=line-too-long
        # see https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=execution-about-job-states
        return {
            "RUN": SmartSimStatus.STATUS_RUNNING,
            "PSUSP": SmartSimStatus.STATUS_PAUSED,
            "USUSP": SmartSimStatus.STATUS_PAUSED,
            "SSUSP": SmartSimStatus.STATUS_PAUSED,
            "PEND": SmartSimStatus.STATUS_PAUSED,
            "DONE": SmartSimStatus.STATUS_COMPLETED,
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
                    SmartSimStatus.STATUS_COMPLETED
                    if returncode == 0
                    else SmartSimStatus.STATUS_FAILED
                )
            else:
                smartsim_status = SmartSimStatus.STATUS_COMPLETED
                returncode = 0
        else:
            smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )


class LSFJsrunStepInfo(StepInfo):  # cov-lsf
    @property
    def mapping(self) -> t.Dict[str, SmartSimStatus]:
        # pylint: disable=line-too-long
        # see https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=execution-about-job-states
        return {
            "Killed": SmartSimStatus.STATUS_COMPLETED,
            "Running": SmartSimStatus.STATUS_RUNNING,
            "Queued": SmartSimStatus.STATUS_PAUSED,
            "Complete": SmartSimStatus.STATUS_COMPLETED,
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
                    SmartSimStatus.STATUS_COMPLETED
                    if returncode == 0
                    else SmartSimStatus.STATUS_FAILED
                )
            else:
                smartsim_status = SmartSimStatus.STATUS_COMPLETED
                returncode = 0
        else:
            smartsim_status = self._get_smartsim_status(status, returncode)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )
