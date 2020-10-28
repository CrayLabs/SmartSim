import psutil
from ..shell import execute_async_cmd
from .localStep import LocalStep
from ..taskManager import TaskManager
from ...error.errors import LauncherError, SSUnsupportedError
from ..stepInfo import StepInfo

from ...utils import get_logger
logger = get_logger(__name__)


class LocalLauncher:
    """Launcher used for spawning proceses on a localhost machine.

    The Local Launcher is primarily used for testing and prototyping
    purposes, this launcher does not have the same capability as the
    launchers that inherit from the SmartSim launcher base class as those
    launcher interact with the workload manager.

    All jobs will be launched serially and will not be able to be queried
    through the controller interface like jobs submitted to a workload
    manager like Slurm.
    """

    def __init__(self):
        self.task_manager = TaskManager()

    def create_step(self, name, run_settings, multi_prog=False):
        """Create a job step to launch an entity locally

        :param name: name of the step to be launch, usually entity.name
        :type name: str
        :param run_settings: smartsim run_settings for an entity
        :type run_settings: dict
        :param multi_prog: create a multi-program step (not supported),
                           but retained for consistency with other launchers
        :type multi_prog: bool, optional
        :raises SSUnsupportedError: if multi_prog is True
        :return: Step object
        """
        if multi_prog:
            raise SSUnsupportedError(
                "Local Launcher does not support multiple program jobs"
            )
        step = LocalStep(run_settings)
        return step

    def get_step_status(self, step_id):
        # get status from task manager
        psutil_status, psutil_rc = self._get_process_status(step_id)
        if self.task_manager.check_error(step_id):
            returncode, out, err = self.task_manager.get_task_history(step_id)
            return StepInfo(psutil_status, returncode, out, err)
        else:
            return StepInfo(psutil_status, psutil_rc)

    def get_step_update(self, step_ids):
        """Get status updates of all steps at once

        :param step_ids: list of step_ids (str)
        :type step_ids: list
        :return: list of StepInfo for update
        :rtype: list
        """
        # these return relatively quick, no need to do anything
        # special here like slurm
        updates = [self.get_step_status(step_id) for step_id in step_ids]
        return updates

    def get_step_nodes(self, step_id):
        """Return the address of nodes assigned to the step

        :return: a list containing the local host address
        """
        return ["127.0.0.1"]

    def run(self, step):
        """Run a local step created by this launcher. Utilize the shell
           library to execute the command with a Popen. Output and error
           files will be written to the entity path.

        :param step: LocalStep instance to run
        :type step: LocalStep
        """
        if not self.task_manager.actively_monitoring:
            self.task_manager.start()

        out_file = open(step.run_settings["out_file"], "w+")
        err_file = open(step.run_settings["err_file"], "w+")
        cmd = step.build_cmd()
        task = execute_async_cmd(cmd, step.cwd, env=step.env, out=out_file, err=err_file)
        self.task_manager.add_task(task, str(task.pid))
        return str(task.pid)

    def stop(self, step_id):
        self.task_manager.remove_task(step_id)
        rc, _, _ = self.task_manager.get_task_history(step_id)
        status = StepInfo("cancelled by user", rc)
        return status

    def is_finished(self, step_id):
        # see https://github.com/giampaolo/psutil/blob/master/psutil/_common.py
        try:
            process = psutil.Process(int(step_id))
            return not process.is_running()
        except psutil.NoSuchProcess:
            return True

    def _get_process_status(self, step_id):
        try:
            task = self.task_manager[step_id]
            return task.status, task.returncode
        # either task manager removed the task already
        # or task has died while still in task manager
        except (psutil.NoSuchProcess, KeyError):
            returncode, _, _ = self.task_manager.get_task_history(step_id)
            if returncode != 0:
                return "failed", returncode
            else:
                return "completed", returncode

    def __str__(self):
        return "local"