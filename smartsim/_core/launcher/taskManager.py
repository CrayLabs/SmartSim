# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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

import time
from subprocess import PIPE
from threading import RLock, Thread

import psutil

from ...error import LauncherError
from ...log import get_logger
from .util.shell import execute_async_cmd, execute_cmd
from ..utils.helpers import check_dev_log_level

logger = get_logger(__name__)
verbose_tm = check_dev_log_level()


TM_INTERVAL = 1


class TaskManager:
    """The Task Manager watches the subprocesses launched through
    the asyncronous shell interface. Each task is a wrapper
    around the Popen/Process instance.

    The Task Managers polls processes on TM_INTERVAL
    and detects job failure and completion. Upon termination, the
    task returncode, output, and error are added to the task history.

    When a launcher uses the task manager to start a task, the task
    is either managed (by a WLM) or unmanaged (meaning not managed by
    a WLM). In the latter case, the Task manager is responsible for the
    lifecycle of the process.
    """

    def __init__(self):
        """Initialize a task manager thread."""
        self.actively_monitoring = False
        self.task_history = dict()
        self.tasks = []
        self._lock = RLock()

    def start(self):
        """Start the task manager thread

        The TaskManager is run as a daemon thread meaning
        that it will die when the main thread dies.
        """
        monitor = Thread(name="TaskManager", daemon=True, target=self.run)
        monitor.start()

    def run(self):
        """Start monitoring Tasks"""

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
        """Start a task managed by the TaskManager

        This is an "unmanaged" task, meaning it is NOT managed
        by a workload manager

        :param cmd_list: command to run
        :type cmd_list: list[str]
        :param cwd: current working directory
        :type cwd: str
        :param env: environment to launch with
        :type env: dict[str, str], optional
        :param out: output file, defaults to PIPE
        :type out: file, optional
        :param err: error file, defaults to PIPE
        :type err: file, optional
        :return: task id
        :rtype: int
        """
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
        """Start a task not managed by the TaskManager

        This method is used by launchers to launch managed tasks
        meaning that they ARE managed by a WLM.

        This is primarily used for batch job launches

        :param cmd_list: command to run
        :type cmd_list: list[str]
        :param cwd: current working directory
        :type cwd: str
        :param env: environment to launch with
        :type env: dict[str, str], optional
        :param timeout: time to wait, defaults to None
        :type timeout: int, optional
        :return: returncode, output, and err
        :rtype: int, str, str
        """
        returncode, out, err = execute_cmd(cmd_list, cwd=cwd, env=env, timeout=timeout)
        if verbose_tm:
            logger.debug("Ran and waited on task")
        return returncode, out, err

    def add_existing(self, task_id):
        """Add existing task to be managed by the TaskManager

        :param task_id: task id of existing task
        :type task_id: int
        :raises LauncherError: If task cannot be found
        """
        self._lock.acquire()
        try:
            process = psutil.Process(pid=task_id)
            task = Task(process)
            self.tasks.append(task)
            self.task_history[task.pid] = (None, None, None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            raise LauncherError(f"Process provided {task_id} does not exist") from None
        finally:
            self._lock.release()

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
        """Get the update of a task

        :param task_id: task id
        :type task_id: str
        :return: status, returncode, output, error
        :rtype: str, int, str, str
        """
        self._lock.acquire()
        try:
            rc, out, err = self.task_history[task_id]
            # has to be == None because rc can be 0
            if rc == None:
                try:
                    task = self[task_id]
                    return task.status, rc, out, err
                # removed forcefully either by OS or us, no returncode set
                # either way, job has completed and we won't have returncode
                # Usually hits when jobs last less then the TM_INTERVAL
                except (KeyError, psutil.NoSuchProcess):
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
        """Add a task to the task history

        Add a task to record its future returncode, output and error

        :param task_id: id of the task
        :type task_id: str
        :param returncode: returncode
        :type returncode: int
        :param out: output, defaults to None
        :type out: str, optional
        :param err: output, defaults to None
        :type err: str, optional
        """
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
    def __init__(self, process):
        """Initialize a task

        :param process: Popen object
        :type process: psutil.Popen
        """
        self.process = process
        self.pid = str(self.process.pid)

    def check_status(self):
        """Ping the job and return the returncode if finished

        :return: returncode if finished otherwise None
        :rtype: int
        """
        if self.owned:
            return self.process.poll()
        # we can't manage Processed we don't own
        # have to rely on .kill() to stop.
        return self.returncode

    def get_io(self):
        """Get the IO from the subprocess

        :return: output and error from the Popen
        :rtype: str, str
        """
        # Process class does not implement communicate
        if not self.owned:
            return None, None
        output, error = self.process.communicate()
        if output:
            output = output.decode("utf-8")
        if error:
            error = error.decode("utf-8")
        return output, error

    def kill(self, timeout=10):
        """Kill the subprocess and all childen"""

        def kill_callback(proc):
            logger.debug(f"Process terminated with kill {proc.pid}")

        children = self.process.children(recursive=True)
        children.append(self.process)  # add parent process to be killed

        for child in children:
            child.kill()

        _, alive = psutil.wait_procs(children, timeout=timeout, callback=kill_callback)
        if alive:
            for proc in alive:
                logger.warning(f"Unable to kill emitted process {proc.pid}")

    def terminate(self, timeout=10):
        """Terminate a this process and all children.

        :param timeout: time to wait for task death, defaults to 10
        :type timeout: int, optional
        """

        def terminate_callback(proc):
            logger.debug(f"Cleanly terminated task {proc.pid}")

        children = self.process.children(recursive=True)
        children.append(self.process)  # add parent process to be killed

        # try SIGTERM first for clean exit
        for child in children:
            logger.debug(child)
            child.terminate()

        # wait for termination
        _, alive = psutil.wait_procs(
            children, timeout=timeout, callback=terminate_callback
        )

        if alive:
            logger.debug(f"SIGTERM failed, using SIGKILL")
            self.process.kill()

    def wait(self):
        self.process.wait()

    @property
    def returncode(self):
        if self.owned:
            return self.process.returncode
        if self.is_alive:
            return None
        return 0

    @property
    def is_alive(self):
        return self.process.is_running()

    @property
    def status(self):
        return self.process.status()

    @property
    def owned(self):
        if isinstance(self.process, psutil.Popen):
            return True
        return False
