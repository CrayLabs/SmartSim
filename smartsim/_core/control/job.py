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

import time
import typing as t

from ...entity import SmartSimEntity, EntityList
from ...status import STATUS_NEW


class Job:
    """Keep track of various information for the controller.
    In doing so, continuously add various fields of information
    that is queryable by the user through interface methods in
    the controller class.
    """

    def __init__(
        self,
        job_name: str,
        job_id: t.Optional[str],
        entity: t.Union[SmartSimEntity, EntityList],
        launcher: str,
        is_task: bool,
    ) -> None:
        """Initialize a Job.

        :param job_name: Name of the job step
        :type job_name: str
        :param job_id: The id associated with the job
        :type job_id: str
        :param entity: The SmartSim entity(list) associated with the job
        :type entity: SmartSimEntity | EntityList
        :param launcher: Launcher job was started with
        :type launcher: str
        :param is_task: process monitored by TaskManager (True) or the WLM (True)
        :type is_task: bool
        """
        self.name = job_name
        self.jid = job_id
        self.entity = entity
        self.status = STATUS_NEW
        # status before smartsim status mapping is applied
        self.raw_status: t.Optional[str] = None
        self.returncode: t.Optional[int] = None
        # output is only populated if it's system related (e.g. cmd failed immediately)
        self.output: t.Optional[str] = None
        self.error: t.Optional[str] = None  # same as output
        self.hosts: t.List[str] = []  # currently only used for DB jobs
        self.launched_with = launcher
        self.is_task = is_task
        self.start_time = time.time()
        self.history = History()

    @property
    def ename(self) -> str:
        """Return the name of the entity this job was created from"""
        return self.entity.name

    def set_status(
        self,
        new_status: str,
        raw_status: str,
        returncode: t.Optional[int],
        error: t.Optional[str] = None,
        output: t.Optional[str] = None,
    ) -> None:
        """Set the status  of a job.

        :param new_status: The new status of the job
        :type new_status: str
        :param returncode: The return code for the job
        :type return_code: str
        """
        self.status = new_status
        self.raw_status = raw_status
        self.returncode = returncode
        self.error = error
        self.output = output

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def record_history(self) -> None:
        """Record the launching history of a job."""
        self.history.record(self.jid, self.status, self.returncode, self.elapsed)

    def reset(
        self, new_job_name: str, new_job_id: t.Optional[str], is_task: bool
    ) -> None:
        """Reset the job in order to be able to restart it.

        :param new_job_name: name of the new job step
        :type new_job_name: str
        :param new_job_id: new job id to launch under
        :type new_job_id: int
        :param is_task: process monitored by TaskManager (True) or the WLM (True)
        :type is_task: bool
        """
        self.name = new_job_name
        self.jid = new_job_id
        self.status = STATUS_NEW
        self.returncode = None
        self.output = None
        self.error = None
        self.hosts = []
        self.is_task = is_task
        self.start_time = time.time()
        self.history.new_run()

    def error_report(self) -> str:
        """A descriptive error report based on job fields

        :return: error report for display in terminal
        :rtype: str
        """
        warning = f"{self.ename} failed. See below for details \n"
        if self.error:
            warning += (
                f"{self.entity.type} {self.ename} produced the following error \n"
            )
            warning += f"Error: {self.error} \n"
        if self.output:
            warning += f"Output: {self.output} \n"
        warning += f"Job status at failure: {self.status} \n"
        warning += f"Launcher status at failure: {self.raw_status} \n"
        warning += f"Job returncode: {self.returncode} \n"
        warning += f"Error and output file located at: {self.entity.path}"
        return warning

    def __str__(self) -> str:
        """Return user-readable string of the Job

        :returns: A user-readable string of the Job
        :rtype: str
        """
        if self.jid:
            job = "{}({}): {}"
            return job.format(self.ename, self.jid, self.status)

        job = "{}: {}"
        return job.format(self.ename, self.status)


class History:
    """History of a job instance. Holds various attributes based
    on the previous launches of a job.
    """

    def __init__(self, runs: int = 0) -> None:
        """Init a history object for a job

        :param runs: number of runs so far, defaults to 0
        :type runs: int, optional
        """
        self.runs = runs
        self.jids: t.Dict[int, t.Optional[str]] = {}
        self.statuses: t.Dict[int, str] = {}
        self.returns: t.Dict[int, t.Optional[int]] = {}
        self.job_times: t.Dict[int, float] = {}

    def record(
        self,
        job_id: t.Optional[str],
        status: str,
        returncode: t.Optional[int],
        job_time: float,
    ) -> None:
        """record the history of a job"""
        self.jids[self.runs] = job_id
        self.statuses[self.runs] = status
        self.returns[self.runs] = returncode
        self.job_times[self.runs] = job_time

    def new_run(self) -> None:
        """increment run total"""
        self.runs += 1
