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

import pathlib
import time
import typing as t
from dataclasses import dataclass

from ...entity import EntitySequence, SmartSimEntity
from ...status import SmartSimStatus


@dataclass(frozen=True)
class _JobKey:
    step_id: str
    """The process id of an unmanaged task"""
    task_id: str
    """The task id of a managed task"""


class JobEntity:
    """An entity containing the minimum API required for a job processed
    in the JobManager that is also supported by the telemetry monitor
    """

    def __init__(self) -> None:
        self.name: str = ""
        """The entity name"""
        self.path: str = ""
        """The root path for entity output files"""
        self.step_id: str = ""
        """The process id of an unmanaged task"""
        self.task_id: str = ""
        """The task id of a managed task"""
        self.type: str = ""
        """The type of the associated `SmartSimEntity`"""
        self.timestamp: int = 0
        """The timestamp when the entity was created"""
        self.status_dir: str = ""
        """The path configured by the experiment for the entities telemetry output"""
        self.telemetry_on: bool = False
        """"Boolean indicating if optional telemetry is enabled for the entity"""
        self.collectors: t.Dict[str, str] = {}
        """A mapping of collectors enabled for the entity"""
        self.config: t.Dict[str, str] = {}
        """Telemetry configuration supplied by the experiment"""
        self._is_complete: bool = False
        """Toggle indicating if the entity has completed execution"""

    @property
    def is_db(self) -> bool:
        """Returns `True` if the entity represents a database or database shard"""
        return self.type in ["orchestrator", "dbnode"]

    @property
    def is_managed(self) -> bool:
        """Returns `True` if the entity is managed by a workload manager"""
        return bool(self.step_id)

    @property
    def key(self) -> _JobKey:
        """Return a `_JobKey` that identifies an entity.
        NOTE: not guaranteed to be unique over time due to reused process IDs"""
        return _JobKey(self.step_id, self.task_id)

    @property
    def is_complete(self) -> bool:
        """Returns `True` if the entity has completed execution"""
        return self._is_complete

    def check_completion_status(self) -> None:
        """Check for telemetry outputs indicating the entity has completed
        TODO: determine correct location to avoid exposing telemetry
        implementation details into `JobEntity`
        """
        # avoid touching file-system if not necessary
        if self._is_complete or not self.is_db:
            return

        # status telemetry is tracked in JSON files written to disk. look
        # for a corresponding `stop` event in the entity status directory
        state_file = pathlib.Path(self.status_dir) / "stop.json"
        if state_file.exists():
            self._is_complete = True

    @staticmethod
    def _deserialize_db_metadata(
        entity_dict: t.Dict[str, t.Any], entity: "JobEntity"
    ) -> None:
        """Set properties that are specific to databases and db nodes

        :param entity_dict: The raw dictionary deserialized from manifest JSON
        :type entity_dict: Dict[str, Any]
        :param entity: The entity instance to modify if the entity is a database
        :type entity: JobEntity"""
        if entity.is_db:
            # add collectors if they're configured to be enabled in the manifest
            entity.collectors = {
                "client": entity_dict.get("client_file", ""),
                "client_count": entity_dict.get("client_count_file", ""),
                "memory": entity_dict.get("memory_file", ""),
            }

            entity.telemetry_on = any(entity.collectors.values())
            entity.config["host"] = entity_dict.get("hostname", "")
            entity.config["port"] = entity_dict.get("port", "")

    @staticmethod
    def from_manifest(
        entity_type: str, entity_dict: t.Dict[str, t.Any], exp_dir: str
    ) -> "JobEntity":
        """Deserialize a `JobEntity` instance from a dictionary deserialized
        from manifest JSON

        :param entity_dict: The raw dictionary deserialized from manifest JSON
        :type entity_dict: Dict[str, Any]
        :param entity_type: The type of the associated `SmartSimEntity`
        :type entity_type: str
        :param exp_dir: The path to the experiment working directory
        :type exp_dir: str"""
        entity = JobEntity()
        metadata = entity_dict["telemetry_metadata"]
        status_dir = pathlib.Path(metadata.get("status_dir"))

        # all entities contain shared properties that identify the task
        entity.type = entity_type
        entity.name = entity_dict["name"]
        entity.step_id = str(metadata.get("step_id") or "")
        entity.task_id = str(metadata.get("task_id") or "")
        entity.timestamp = int(entity_dict.get("timestamp", "0"))
        entity.path = str(exp_dir)
        entity.status_dir = str(status_dir)

        JobEntity._deserialize_db_metadata(entity_dict, entity)

        return entity


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
        entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity], JobEntity],
        launcher: str,
        is_task: bool,
    ) -> None:
        """Initialize a Job.

        :param job_name: Name of the job step
        :type job_name: str
        :param job_id: The id associated with the job
        :type job_id: str
        :param entity: The SmartSim entity(list) associated with the job
        :type entity: SmartSimEntity | EntitySequence | JobEntity
        :param launcher: Launcher job was started with
        :type launcher: str
        :param is_task: process monitored by TaskManager (True) or the WLM (True)
        :type is_task: bool
        """
        self.name = job_name
        self.jid = job_id
        self.entity = entity
        self.status = SmartSimStatus.STATUS_NEW
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
        new_status: SmartSimStatus,
        raw_status: str,
        returncode: t.Optional[int],
        error: t.Optional[str] = None,
        output: t.Optional[str] = None,
    ) -> None:
        """Set the status  of a job.

        :param new_status: The new status of the job
        :type new_status: SmartSimStatus
        :param raw_status: The raw status of the launcher
        :type raw_status: str
        :param returncode: The return code for the job
        :type return_code: int
        :param error: Content produced by stderr
        :type error: str
        :param output: Content produced by stdout
        :type output: str
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
        self.status = SmartSimStatus.STATUS_NEW
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
        self.statuses: t.Dict[int, SmartSimStatus] = {}
        self.returns: t.Dict[int, t.Optional[int]] = {}
        self.job_times: t.Dict[int, float] = {}

    def record(
        self,
        job_id: t.Optional[str],
        status: SmartSimStatus,
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
