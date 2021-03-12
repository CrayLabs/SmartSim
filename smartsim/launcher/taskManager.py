import os
import time
from threading import RLock, Thread
from subprocess import PIPE
import numpy as np
import psutil
from .util.shell import execute_async_cmd, execute_cmd
from ..constants import TM_INTERVAL
from ..utils import get_logger

logger = get_logger(__name__)

try:
    level = os.environ["SMARTSIM_LOG_LEVEL"]
    verbose_tm = True if level == "developer" else False
except KeyError:
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
            logger.debug("Starting Task Manager")

        self.actively_monitoring = True
        while self.actively_monitoring:
            time.sleep(TM_INTERVAL)

            for task in self.tasks:
                returncode = task.check_status()  # poll and set returncode
                # has to be != None because returncode can be 0
                if returncode is not None:
                    output, error = task.get_io()
                    self.add_task_history(task.pid, returncode, output, error)
                    self.remove_task(task.pid)

            if len(self) == 0:
                self.actively_monitoring = False
                if verbose_tm:
                    logger.debug("Sleeping, no tasks to monitor")

    def start_task(self, cmd_list, cwd, env=None, out=PIPE, err=PIPE):
        self._lock.acquire()
        try:
            proc = execute_async_cmd(cmd_list, cwd, env=env, out=out, err=err)
            task = Task(proc)
            if verbose_tm:
                logger.debug(f"Starting Task {task.pid}")
            self.tasks.append(task)
            self.task_history[task.pid] = (None, None, None)
            return task.pid

        finally:
            self._lock.release()

    def start_and_wait(self, cmd_list, cwd, env=None, timeout=None):
        returncode, out, err = execute_cmd(cmd_list, cwd=cwd, env=env, timeout=timeout)
        if verbose_tm:
            logger.debug("Ran and waited on task")
        return returncode, out, err

    def remove_task(self, task_id):
        """Remove a task from the TaskManager

        :param task_id: id of the task to remove
        :type task_id: str
        """
        self._lock.acquire()
        if verbose_tm:
            logger.debug(f"Removing Task {task_id}")
        try:
            task = self[task_id]
            if task.is_alive:
                task.kill()
                task.wait()
                returncode = task.check_status()
                out, err = task.get_io()
                self.add_task_history(task_id, returncode, out, err)
            self.tasks.remove(task)
        except psutil.NoSuchProcess:
            logger.debug("Failed to kill a task during removal")
        except KeyError:
            logger.debug("Failed to remove a task, task was already removed")
        finally:
            self._lock.release()

    def get_task_update(self, task_id):
        self._lock.acquire()
        try:
            rc, out, err = self.task_history[task_id]
            # has to be == None because rc can be 0
            if rc == None:
                try:
                    task = self[task_id]
                    return task.status, rc, out, err
                # removed forcefully either by OS or us, no returncode set
                except KeyError:
                    return "Completed", rc, out, err
            # process has completed, status set manually as we don't
            # save task statuses during runtime.
            else:
                if rc != 0:
                    return "Failed", rc, out, err
                return "Completed", rc, out, err
        finally:
            self._lock.release()

    def add_task_history(self, task_id, returncode, out=None, err=None):
        self.task_history[task_id] = (returncode, out, err)

    def __getitem__(self, task_id):
        self._lock.acquire()
        try:
            for task in self.tasks:
                if task.pid == task_id:
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

    def __init__(self, popen_process):
        """Initialize a task

        :param popen_process: Popen object
        :type popen_process: psutil.Popen
        """
        self.process = popen_process
        self.pid = str(self.process.pid)

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

    def wait(self):
        self.process.wait()

    @property
    def returncode(self):
        return self.process.returncode

    @property
    def is_alive(self):
        return self.process.is_running()

    @property
    def status(self):
        return self.process.status()

