import psutil

from ..constants import (
    SMARTSIM_STATUS,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PAUSED,
    STATUS_RUNNING,
)


class StepInfo:
    def __init__(
        self, status="", launcher_status="", returncode=None, output=None, error=None
    ):
        self.status = status
        self.launcher_status = launcher_status
        self.returncode = returncode
        self.output = output
        self.error = error


class LocalStepInfo(StepInfo):

    # see https://github.com/giampaolo/psutil/blob/master/psutil/_pslinux.py
    # see https://github.com/giampaolo/psutil/blob/master/psutil/_common.py
    mapping = {
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

    def __init__(self, status="", returncode=None, output=None, error=None):
        smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )

    def _get_smartsim_status(self, status):
        if status in SMARTSIM_STATUS:
            return SMARTSIM_STATUS[status]
        if status in self.mapping:
            return self.mapping[status]
        # we don't know what happened so return failed to be safe
        return STATUS_FAILED


class SlurmStepInfo(StepInfo):

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

    def __init__(self, status="", returncode=None, output=None, error=None):
        smartsim_status = self._get_smartsim_status(status)
        super().__init__(
            smartsim_status, status, returncode, output=output, error=error
        )

    def _get_smartsim_status(self, status):
        if status in SMARTSIM_STATUS:
            return SMARTSIM_STATUS[status]
        if status in self.mapping:
            return self.mapping[status]
        # we don't know what happened so return failed to be safe
        return STATUS_FAILED
