# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import psutil
import typing as t

from ...status import (
    SMARTSIM_STATUS,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PAUSED,
    STATUS_RUNNING,
)


class StepInfo:
    def __init__(
        self,
        status: str = "",
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
        info_str = f"Status: {self.status}"
        info_str += f" | Launcher Status {self.launcher_status}"
        info_str += f" | Returncode {str(self.returncode)}"
        return info_str

    @property
    def mapping(self) -> t.Dict[str, str]:
        raise NotImplementedError

    def _get_smartsim_status(
        self, status: str, returncode: t.Optional[int] = None
    ) -> str:
        """
        Map the status of the WLM step to a smartsim-specific status
        """
        if status in SMARTSIM_STATUS:
            return SMARTSIM_STATUS[status]

        if status in self.mapping and returncode in [None, 0]:
            return self.mapping[status]

        return STATUS_FAILED


class UnmanagedStepInfo(StepInfo):
    @property
    def mapping(self) -> t.Dict[str, str]:
        # see https://github.com/giampaolo/psutil/blob/master/psutil/_pslinux.py
        # see https://github.com/giampaolo/psutil/blob/master/psutil/_common.py
        return {
            psutil.STATUS_RUNNING: STATUS_RUNNING,
            psutil.STATUS_SLEEPING: STATUS_RUNNING,  # sleeping thread is still alive
            psutil.STATUS_WAKING: STATUS_RUNNING,
            psutil.STATUS_DISK_SLEEP: STATUS_RUNNING,
            psutil.STATUS_DEAD: STATUS_FAILED,
            psutil.STATUS_TRACING_STOP: STATUS_PAUSED,
            psutil.STATUS_WAITING: STATUS_PAUSED,
            psutil.STATUS_STOPPED: STATUS_PAUSED,
            psutil.STATUS_LOCKED: STATUS_PAUSED,
            psutil.STATUS_PARKED: STATUS_PAUSED,
            psutil.STATUS_IDLE: STATUS_PAUSED,
            psutil.STATUS_ZOMBIE: STATUS_COMPLETED,
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
        "RUNNING": STATUS_RUNNING,
        "CONFIGURING": STATUS_RUNNING,
        "STAGE_OUT": STATUS_RUNNING,
        "COMPLETED": STATUS_COMPLETED,
        "DEADLINE": STATUS_COMPLETED,
        "TIMEOUT": STATUS_COMPLETED,
        "BOOT_FAIL": STATUS_FAILED,
        "FAILED": STATUS_FAILED,
        "NODE_FAIL": STATUS_FAILED,
        "OUT_OF_MEMORY": STATUS_FAILED,
        "CANCELLED": STATUS_CANCELLED,
        "CANCELLED+": STATUS_CANCELLED,
        "REVOKED": STATUS_CANCELLED,
        "PENDING": STATUS_PAUSED,
        "PREEMPTED": STATUS_PAUSED,
        "RESV_DEL_HOLD": STATUS_PAUSED,
        "REQUEUE_FED": STATUS_PAUSED,
        "REQUEUE_HOLD": STATUS_PAUSED,
        "REQUEUED": STATUS_PAUSED,
        "RESIZING": STATUS_PAUSED,
        "SIGNALING": STATUS_PAUSED,
        "SPECIAL_EXIT": STATUS_PAUSED,
        "STOPPED": STATUS_PAUSED,
        "SUSPENDED": STATUS_PAUSED,
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
    def mapping(self) -> t.Dict[str, str]:
        # pylint: disable=line-too-long
        # see http://nusc.nsu.ru/wiki/lib/exe/fetch.php/doc/pbs/PBSReferenceGuide19.2.1.pdf#M11.9.90788.PBSHeading1.81.Job.States
        return {
            "R": STATUS_RUNNING,
            "B": STATUS_RUNNING,
            "H": STATUS_PAUSED,
            "M": STATUS_PAUSED,  # Actually means that it was moved to another server,
            # TODO: understand what this implies
            "Q": STATUS_PAUSED,
            "S": STATUS_PAUSED,
            "T": STATUS_PAUSED,  # This means in transition, see above for comment
            "U": STATUS_PAUSED,
            "W": STATUS_PAUSED,
            "E": STATUS_COMPLETED,
            "F": STATUS_COMPLETED,
            "X": STATUS_COMPLETED,
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
                smartsim_status = "Completed" if returncode == 0 else "Failed"
            else:
                # if PBS job history isnt available, and job isnt in queue
                smartsim_status = "Completed"
                returncode = 0
        else:
            smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )


class CobaltStepInfo(StepInfo):  # cov-cobalt
    @property
    def mapping(self) -> t.Dict[str, str]:
        return {
            "running": STATUS_RUNNING,
            "queued": STATUS_PAUSED,
            "starting": STATUS_PAUSED,
            "dep_hold": STATUS_PAUSED,
            "user_hold": STATUS_PAUSED,
            "admin_hold": STATUS_PAUSED,
            "dep_fail": STATUS_FAILED,  # unsure of this one
            "terminating": STATUS_COMPLETED,
            "killing": STATUS_COMPLETED,
            "exiting": STATUS_COMPLETED,
        }

    def __init__(
        self,
        status: str = "",
        returncode: t.Optional[int] = None,
        output: t.Optional[str] = None,
        error: t.Optional[str] = None,
    ) -> None:
        if status == "NOTFOUND":
            # returncode not logged by Cobalt
            # if job has exited the queue then we consider it "completed"
            # this should only be hit in the case when job exits abnormally fast
            smartsim_status = "Completed"
            returncode = 0
        else:
            smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )


class LSFBatchStepInfo(StepInfo):  # cov-lsf
    @property
    def mapping(self) -> t.Dict[str, str]:
        # pylint: disable=line-too-long
        # see https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=execution-about-job-states
        return {
            "RUN": STATUS_RUNNING,
            "PSUSP": STATUS_PAUSED,
            "USUSP": STATUS_PAUSED,
            "SSUSP": STATUS_PAUSED,
            "PEND": STATUS_PAUSED,
            "DONE": STATUS_COMPLETED,
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
                smartsim_status = "Completed" if returncode == 0 else "Failed"
            else:
                smartsim_status = "Completed"
                returncode = 0
        else:
            smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )


class LSFJsrunStepInfo(StepInfo):  # cov-lsf
    @property
    def mapping(self) -> t.Dict[str, str]:
        # pylint: disable=line-too-long
        # see https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=execution-about-job-states
        return {
            "Killed": STATUS_COMPLETED,
            "Running": STATUS_RUNNING,
            "Queued": STATUS_PAUSED,
            "Complete": STATUS_COMPLETED,
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
                smartsim_status = "Completed" if returncode == 0 else "Failed"
            else:
                smartsim_status = "Completed"
                returncode = 0
        else:
            smartsim_status = self._get_smartsim_status(status, returncode)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )
