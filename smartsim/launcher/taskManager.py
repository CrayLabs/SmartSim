import time
from threading import RLock, Thread

import numpy as np
import psutil

from ..constants import TM_INTERVAL
from ..error import SSConfigError
from ..utils import get_env, get_logger

logger = get_logger(__name__)

try:
    level = get_env("SMARTSIM_LOG_LEVEL")
    verbose_tm = True if level == "developer" else False
except SSConfigError:
    verbose_tm = False


class TaskManager:
    """The Task Manager watches the subprocesses launched through
    the asyncronous shell interface. Each task is a wrapper
    around the popen that links it to the id of the Job instance
    it is connected to.

    The Task manager can be optionally added to any launcher interface
    to have greater control over the entities that are launched.

    The task manager connects to the job manager through the launcher
    so as to not break encapsulation between the controller and launcher.

    Each time a status is requested, a launcher can check the task
    manager to ensure that the task is still alive and well.
    """

    def __init__(self):
        """Initialize a task manager thread."""
        self.name = "TaskManager" + "-" + str(np.base_repr(time.time_ns(), 36))
        self.actively_monitoring = False
        self.task_history = dict()
        self.tasks = []
        self._lock = RLock()

    def start(self):
        """Start the task manager thread"""
        monitor = Thread(name=self.name, daemon=True, target=self.run)
        monitor.start()

    def run(self):
        """Start the loop that continually checks tasks for status."""
        global verbose_tm
        if verbose_tm:
            logger.debug(f"Starting Task Manager")

        self.actively_monitoring = True
        while self.actively_monitoring:
            time.sleep(TM_INTERVAL)

            for task in self.tasks:
                returncode = task.check_status()  # poll and set returncode
                # has to be != None because returncode can be 0
                if returncode is not None:
                    output, error = task.get_io()
                    self.add_task_history(task.step_id, returncode, output, error)
                    self.remove_task(task.step_id)

            if len(self) == 0:
                self.actively_monitoring = False
                if verbose_tm:
                    logger.debug(f"Sleeping, no tasks to monitor")

    def add_task(self, popen_process, step_id):
        """Create and add a task to the TaskManager

        :param popen_process: Popen object
        :type popen_process: psutil.Popen
        :param step_id: id gleaned from the launcher
        :type step_id: str
        """
        # should all be atomic
        task = Task(popen_process, step_id)
        if verbose_tm:
            logger.debug(f"Adding Task {step_id}")
        self.tasks.append(task)
        self.task_history[step_id] = (None, None, None)

    def remove_task(self, step_id):
        """Remove a task from the TaskManager

        :param step_id: step_id of the task to remove
        :type step_id: str
        """
        self._lock.acquire()
        if verbose_tm:
            logger.debug(f"Removing Task {step_id}")
        try:
            task = self[step_id]
            if task.is_alive:
                task.kill()
                returncode = task.check_status()
                self.add_task_history(task.step_id, returncode)
            self.tasks.remove(task)
        except psutil.NoSuchProcess:
            logger.debug("Failed to kill a task during removal")
        except KeyError:
            logger.debug("Failed to remove a task, task was already removed")
        finally:
            self._lock.release()

    def check_error(self, step_id):
        """Check to see if the job has an error

        :param step_id: step_id of the job
        :type step_id: str
        """
        self._lock.acquire()
        try:
            history = self.task_history[step_id]
            # task is still running
            if history[0] is None:
                return False
            # task recorded non-zero exit code
            if history[0] != 0:
                return True
            return False
        # Task hasnt been added yet (unlikely)
        except KeyError:
            return False
        finally:
            self._lock.release()

    def get_task_history(self, step_id):
        # dictionary access is atomic
        history = self.task_history[step_id]
        return history

    def add_task_history(self, step_id, returncode, out=None, err=None):
        # setting dictionary attributes is atomic
        self.task_history[step_id] = (returncode, out, err)

    def __getitem__(self, step_id):
        self._lock.acquire()
        try:
            for task in self.tasks:
                if task.step_id == step_id:
                    return task
            raise KeyError
        finally:
            self._lock.release()

    def __len__(self):
        self._lock.acquire()
        try:
            return len(self.tasks)
        finally:
            self._lock.release()


class Task:
    """A Task is a wrapper around a Popen object that includes a reference
    to the Job id created by the launcher. For the local launcher this
    will just be the pid of the Popen object
    """

    def __init__(self, popen_process, step_id):
        """Initialize a task

        :param popen_process: Popen object
        :type popen_process: psutil.Popen
        :param step_id: Id from the launcher
        :type step_id: str
        """
        self.process = popen_process
        self.step_id = step_id  # dependant on the launcher type

    def has_piped_io(self):
        """When jobs are spawned using the command server they
           will not have any IO as you cannot serialize a Popen
           object with open PIPEs

        :return: boolean for if Popen has PIPEd IO
        :rtype: bool
        """
        if self.process.stdout or self.process.stderr:
            return True
        return False

    def check_status(self):
        """Ping the job and return the returncode if finished

        :return: returncode if finished otherwise None
        :rtype: int
        """
        return self.process.poll()

    def get_io(self):
        """Get the IO from the subprocess

        :return: output and error from the Popen
        :rtype: str, str
        """
        output, error = self.process.communicate()
        if output:
            output = output.decode("utf-8")
        if error:
            error = error.decode("utf-8")
        return output, error

    def kill(self):
        """Kill the subprocess"""
        self.process.kill()

    @property
    def pid(self):
        return str(self.process.pid)

    @property
    def returncode(self):
        return self.process.returncode

    @property
    def is_alive(self):
        return self.process.is_running()

    @property
    def status(self):
        return self.process.status()

    def __repr__(self):
        return self.step_id
