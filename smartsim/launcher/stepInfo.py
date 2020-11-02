import psutil
from ..constants import STATUS_CANCELLED, STATUS_COMPLETED, STATUS_FAILED
from ..constants import STATUS_NEW, STATUS_PAUSED, STATUS_RUNNING, SMARTSIM_STATUS


class StepInfo:

    def __init__(self, status="", returncode=None, output=None, error=None):
        self.status = status
        self.returncode = returncode
        self.output = output
        self.error = error

class LocalStepInfo(StepInfo):

    mapping = {
        psutil.STATUS_RUNNING: STATUS_RUNNING,
        psutil.STATUS_SLEEPING: STATUS_RUNNING, # sleeping thread is still alive
        psutil.STATUS_WAKING: STATUS_RUNNING,
        psutil.STATUS_DISK_SLEEP: STATUS_RUNNING,
        psutil.STATUS_DEAD: STATUS_FAILED,
        psutil.STATUS_TRACING_STOP: STATUS_PAUSED,
        psutil.STATUS_WAITING: STATUS_PAUSED,
        psutil.STATUS_STOPPED: STATUS_PAUSED,
        psutil.STATUS_LOCKED: STATUS_PAUSED,
        psutil.STATUS_PARKED: STATUS_PAUSED,
        psutil.STATUS_IDLE: STATUS_PAUSED,
        psutil.STATUS_ZOMBIE: STATUS_COMPLETED
    }


    def __init__(self, status="", returncode=None, output=None, error=None):
        smartsim_status = self._get_smartsim_status(status)
        super().__init__(smartsim_status, returncode, output=output, error=error)

    def _get_smartsim_status(self, status):
        # see https://github.com/giampaolo/psutil/blob/master/psutil/_pslinux.py
        # see https://github.com/giampaolo/psutil/blob/master/psutil/_common.py

        if status in SMARTSIM_STATUS:
            return SMARTSIM_STATUS[status]
        elif status in self.mapping:
            return self.mapping[status]
        else:
            # we don't know what happened so return failed to be safe
            return STATUS_FAILED