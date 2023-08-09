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

from __future__ import annotations

import psutil
import time
import typing as t

from subprocess import PIPE
from threading import RLock, Thread

from ...error import LauncherError
from ...log import get_logger
from ..utils.helpers import check_dev_log_level
from .util.shell import execute_async_cmd, execute_cmd

logger = get_logger(__name__)
VERBOSE_TM = check_dev_log_level()  # pylint: disable=invalid-name

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

    def __init__(self) -> None:
        """Initialize a task manager thread."""
        self.actively_monitoring = False
        self.task_history: t.Dict[
            str, t.Tuple[t.Optional[int], t.Optional[str], t.Optional[str]]
        ] = {}
        self.tasks: t.List[Task] = []
        self._lock = RLock()

    def start(self) -> None:
        """Start the task manager thread

        The TaskManager is run as a daemon thread meaning
        that it will die when the main thread dies.
        """
        monitor = Thread(name="TaskManager", daemon=True, target=self.run)
        monitor.start()

    def run(self) -> None:
        """Start monitoring Tasks"""

        if VERBOSE_TM:
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
                if VERBOSE_TM:
                    logger.debug("Sleeping, no tasks to monitor")

    def start_task(
        self,
        cmd_list: t.List[str],
        cwd: str,
        env: t.Optional[t.Dict[str, str]] = None,
        out: int = PIPE,
        err: int = PIPE,
    ) -> str:
        """Start a task managed by the TaskManager

        This is an "unmanaged" task, meaning it is NOT managed
        by a workload manager

        :param cmd_list: command to run
        :type cmd_list: list[str]
        :param cwd: current working directory
        :type cwd: str
        :param env: environment to launch with
        :type env: dict[str, str], optional. If None, calling environment is inherited
        :param out: output file, defaults to PIPE
        :type out: file, optional
        :param err: error file, defaults to PIPE
        :type err: file, optional
        :return: task id
        :rtype: int
        """
        with self._lock:
            proc = execute_async_cmd(cmd_list, cwd, env=env, out=out, err=err)
            task = Task(proc)
            if VERBOSE_TM:
                logger.debug(f"Starting Task {task.pid}")
            self.tasks.append(task)
            self.task_history[task.pid] = (None, None, None)
            return task.pid

    @staticmethod
    def start_and_wait(
        cmd_list: t.List[str],
        cwd: str,
        env: t.Optional[t.Dict[str, str]] = None,
        timeout: t.Optional[int] = None,
    ) -> t.Tuple[int, str, str]:
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
        if VERBOSE_TM:
            logger.debug("Ran and waited on task")
        return returncode, out, err

    def add_existing(self, task_id: int) -> None:
        """Add existing task to be managed by the TaskManager

        :param task_id: task id of existing task
        :type task_id: str
        :raises LauncherError: If task cannot be found
        """
        with self._lock:
            try:
                process = psutil.Process(pid=task_id)
                task = Task(process)
                self.tasks.append(task)
                self.task_history[task.pid] = (None, None, None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                msg = f"Process provided {task_id} does not exist"
                raise LauncherError(msg) from None

    def remove_task(self, task_id: str) -> None:
        """Remove a task from the TaskManager

        :param task_id: id of the task to remove
        :type task_id: str
        """
        with self._lock:
            if VERBOSE_TM:
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

    def get_task_update(
        self, task_id: str
    ) -> t.Tuple[str, t.Optional[int], t.Optional[str], t.Optional[str]]:
        """Get the update of a task

        :param task_id: task id
        :type task_id: str
        :return: status, returncode, output, error
        :rtype: str, int, str, str
        """
        with self._lock:
            try:
                return_code, out, err = self.task_history[task_id]
                # has to be == None because rc can be 0
                if return_code is None:
                    try:
                        task = self[task_id]
                        return task.status, return_code, out, err
                    # removed forcefully either by OS or us, no returncode set
                    # either way, job has completed and we won't have returncode
                    # Usually hits when jobs last less then the TM_INTERVAL
                    except (KeyError, psutil.NoSuchProcess):
                        return "Completed", return_code, out, err

                # process has completed, status set manually as we don't
                # save task statuses during runtime.
                else:
                    if return_code != 0:
                        return "Failed", return_code, out, err
                    return "Completed", return_code, out, err
            except KeyError:
                logger.warning(f"Task {task_id} not found in task history dictionary")

        return "Failed", -1, "", ""

    def add_task_history(
        self,
        task_id: str,
        returncode: t.Optional[int] = None,
        out: t.Optional[str] = None,
        err: t.Optional[str] = None,
    ) -> None:
        """Add a task to the task history

        Add a task to record its future returncode, output and error

        :param task_id: id of the task
        :type task_id: str
        :param returncode: returncode
        :type returncode: int, defaults to None
        :param out: output, defaults to None
        :type out: str, optional
        :param err: output, defaults to None
        :type err: str, optional
        """
        self.task_history[task_id] = (returncode, out, err)

    def __getitem__(self, task_id: str) -> Task:
        with self._lock:
            for task in self.tasks:
                if task.pid == task_id:
                    return task
            raise KeyError

    def __len__(self) -> int:
        with self._lock:
            return len(self.tasks)


class Task:
    def __init__(self, process: psutil.Process) -> None:
        """Initialize a task

        :param process: Popen object
        :type process: psutil.Process
        """
        self.process = process
        self.pid = str(self.process.pid)

    def check_status(self) -> t.Optional[int]:
        """Ping the job and return the returncode if finished

        :return: returncode if finished otherwise None
        :rtype: int
        """
        if self.owned and isinstance(self.process, psutil.Popen):
            poll_result = self.process.poll()
            if poll_result is not None:
                return int(poll_result)
            return None
        # we can't manage Processed we don't own
        # have to rely on .kill() to stop.
        return self.returncode

    def get_io(self) -> t.Tuple[t.Optional[str], t.Optional[str]]:
        """Get the IO from the subprocess

        :return: output and error from the Popen
        :rtype: str, str
        """
        # Process class does not implement communicate
        if not self.owned or not isinstance(self.process, psutil.Popen):
            return None, None

        output, error = self.process.communicate()
        if output:
            output = output.decode("utf-8")
        if error:
            error = error.decode("utf-8")
        return output, error

    def kill(self, timeout: int = 10) -> None:
        """Kill the subprocess and all children"""

        def kill_callback(proc: psutil.Process) -> None:
            logger.debug(f"Process terminated with kill {proc.pid}")

        children = self.process.children(recursive=True)
        children.append(self.process)  # add parent process to be killed

        for child in children:
            child.kill()

        _, alive = psutil.wait_procs(children, timeout=timeout, callback=kill_callback)
        if alive:
            for proc in alive:
                logger.warning(f"Unable to kill emitted process {proc.pid}")

    def terminate(self, timeout: int = 10) -> None:
        """Terminate a this process and all children.

        :param timeout: time to wait for task death, defaults to 10
        :type timeout: int, optional
        """

        def terminate_callback(proc: psutil.Process) -> None:
            logger.debug(f"Cleanly terminated task {proc.pid}")

        children = self.process.children(recursive=True)
        children.append(self.process)  # add parent process to be killed

        # try SIGTERM first for clean exit
        for child in children:
            if VERBOSE_TM:
                logger.debug(child)
            child.terminate()

        # wait for termination
        _, alive = psutil.wait_procs(
            children, timeout=timeout, callback=terminate_callback
        )

        if alive:
            logger.debug("SIGTERM failed, using SIGKILL")
            self.process.kill()

    def wait(self) -> None:
        self.process.wait()

    @property
    def returncode(self) -> t.Optional[int]:
        if self.owned and isinstance(self.process, psutil.Popen):
            if self.process.returncode is not None:
                return int(self.process.returncode)
            return None
        if self.is_alive:
            return None
        return 0

    @property
    def is_alive(self) -> bool:
        return self.process.is_running()

    @property
    def status(self) -> str:
        return self.process.status()

    @property
    def owned(self) -> bool:
        if isinstance(self.process, psutil.Popen):
            return True
        return False
