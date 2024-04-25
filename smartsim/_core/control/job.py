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

import itertools
import pathlib
import time
import typing as t
from dataclasses import dataclass

from ..._core import types as _core_types
from ..._core.utils import helpers as _helpers
from ...entity import EntitySequence, SmartSimEntity
from ...entity.dbnode import DBNode
from ...entity import types as _entity_types
from ...status import TERMINAL_STATUSES, SmartSimStatus

from smartsim._core.utils import network, redis
from smartsim._core.launcher.stepInfo import StepInfo

if t.TYPE_CHECKING:
    from smartsim._core.launcher.launcher import Launcher
    from smartsim._core.launcher.step.step import Step


@dataclass(frozen=True)
class _JobKey:
    """A helper class for creating unique lookup keys within the telemetry
    monitor. These keys are not guaranteed to be unique across experiments,
    only within an experiment (due to process ID re-use by the OS)"""

    step_id: _core_types.StepID
    """The process id of an unmanaged task"""
    task_id: _core_types.TaskID
    """The task id of a managed task"""


class JobEntity:
    """An entity containing run-time SmartSimEntity metadata. The run-time metadata
    is required to perform telemetry collection. The `JobEntity` satisfies the core
    API necessary to use a `JobManager` to manage retrieval of managed step updates.
    """

    def __init__(self) -> None:
        self.name = _entity_types.EntityName("")
        """The entity name"""
        self.path: str = ""
        """The root path for entity output files"""
        self.step_id = _core_types.StepID("")
        """The process id of an unmanaged task"""
        self.task_id = _core_types.TaskID("")
        """The task id of a managed task"""
        self.type: t.Literal[_core_types.TTelmonEntityTypeStr, ""] = ""
        """The type of the associated `SmartSimEntity`"""
        self.timestamp: int = 0
        """The timestamp when the entity was created"""
        self.status_dir: str = ""
        """The path configured by the experiment for the entities telemetry output"""
        self.telemetry_on: bool = False
        """"Flag indicating if optional telemetry is enabled for the entity"""
        self.collectors: t.Dict[str, str] = {}
        """Mapping of collectors enabled for the entity"""
        self.config: t.Dict[str, str] = {}
        """Telemetry configuration supplied by the experiment"""
        self._is_complete: bool = False
        """Flag indicating if the entity has completed execution"""

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
        if self._is_complete:
            return

        # status telemetry is tracked in JSON files written to disk. look
        # for a corresponding `stop` event in the entity status directory
        state_file = pathlib.Path(self.status_dir) / "stop.json"
        if state_file.exists():
            self._is_complete = True

    @staticmethod
    def _map_db_metadata(entity_dict: t.Dict[str, t.Any], entity: "JobEntity") -> None:
        """Map DB-specific properties from a runtime manifest onto a `JobEntity`

        :param entity_dict: The raw dictionary deserialized from manifest JSON
        :type entity_dict: Dict[str, Any]
        :param entity: The entity instance to modify
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
    def _map_standard_metadata(
        entity_type: _core_types.TTelmonEntityTypeStr,
        entity_dict: t.Dict[str, t.Any],
        entity: "JobEntity",
        exp_dir: str,
    ) -> None:
        """Map universal properties from a runtime manifest onto a `JobEntity`

        :param entity_type: The type of the associated `SmartSimEntity`
        :type entity_type: str
        :param entity_dict: The raw dictionary deserialized from manifest JSON
        :type entity_dict: Dict[str, Any]
        :param entity: The entity instance to modify
        :type entity: JobEntity
        :param exp_dir: The path to the experiment working directory
        :type exp_dir: str"""
        metadata = entity_dict["telemetry_metadata"]
        status_dir = pathlib.Path(metadata.get("status_dir"))

        # all entities contain shared properties that identify the task
        entity.type = entity_type
        entity.name = entity_dict["name"]
        entity.step_id = _core_types.StepID(metadata.get("step_id") or "")
        entity.task_id = _core_types.TaskID(metadata.get("task_id") or "")
        entity.timestamp = int(entity_dict.get("timestamp", "0"))
        entity.path = str(exp_dir)
        entity.status_dir = str(status_dir)

    @classmethod
    def from_manifest(
        cls,
        entity_type: _core_types.TTelmonEntityTypeStr,
        entity_dict: t.Dict[str, t.Any],
        exp_dir: str,
    ) -> "JobEntity":
        """Instantiate a `JobEntity` from the dictionary deserialized from manifest JSON

        :param entity_type: The type of the associated `SmartSimEntity`
        :type entity_type: str
        :param entity_dict: The raw dictionary deserialized from manifest JSON
        :type entity_dict: Dict[str, Any]
        :param exp_dir: The path to the experiment working directory
        :type exp_dir: str"""
        entity = JobEntity()

        cls._map_standard_metadata(entity_type, entity_dict, entity, exp_dir)
        cls._map_db_metadata(entity_dict, entity)

        return entity


_TJob = t.TypeVar("_TJob", bound="Job")


class Job:
    """Keep track of various information for the controller.
    In doing so, continuously add various fields of information
    that is queryable by the user through interface methods in
    the controller class.
    """

    def __init__(
        self,
        job_name: _core_types.StepName,
        job_id: _core_types.JobIdType,
        entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity], JobEntity],
        launcher: str,
        is_task: bool,
        update_callback: t.Callable[
            [t.List[_core_types.StepName]],
            t.List[t.Tuple[_core_types.StepName, t.Union[StepInfo, None]]],
        ],
        stop_callback: t.Callable[[_core_types.StepName], StepInfo],
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
        self.launched_with: t.Final = launcher
        self.is_task = is_task
        self.start_time = time.time()
        self.history = History()
        self._update_callback: t.Final = update_callback
        self._stop_callback: t.Final = stop_callback

    @classmethod
    def from_launched_step(
        cls: t.Type[_TJob],
        job_id: _core_types.JobIdType,
        launcher: "Launcher",
        step: "Step",
        entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity], JobEntity],
    ) -> _TJob:
        # XXX: Why is this not the default constructor?? Why does the job not
        #      hold a backref to the step it is tracking (Same question for the
        #      step and the launcher that launched it, tbh)? Why does the job
        #      need to know the entity that it was launched from but the step
        #      does not? Is the job a wrapper around a step or not??
        return cls(
            step.name,
            job_id,
            entity,
            str(launcher),
            not step.managed,
            launcher.get_step_update,
            (
                launcher.stop
                if not isinstance(entity, DBNode)
                else _attempt_graceful_db_node_shutdown(entity, launcher.stop)
                #                                       ^^^^^^
                # XXX: I _HATE_ that I have to pass this reference. At best it
                #      is sloppy, but there is no strong guarentee that the
                #      ``hosts`` attr is defined at job creation time, and
                #      there is no way to populate if the ``Controller`` has
                #      not had time to symlink output files. The best I can do
                #      is delay look up time :(
            ),
        )

    @property
    def ename(self) -> _entity_types.EntityName:
        """Return the name of the entity this job was created from"""
        return self.entity.name

    def _get_update_callback(self) -> t.Callable[
        [t.List[_core_types.StepName]],
        t.List[t.Tuple[_core_types.StepName, t.Union[StepInfo, None]]],
    ]:
        return self._update_callback

    def _get_stop_callback(self) -> t.Callable[[_core_types.StepName], StepInfo]:
        return self._stop_callback

    @classmethod
    def refresh_all(cls, jobs: t.Iterable["Job"]) -> None:
        jobs = _helpers.unique(jobs)
        to_update = _helpers.group_by_map(cls._get_update_callback, jobs)
        for update, stale_jobs in to_update.items():
            stale_job_names = _helpers.unique(job.name for job in stale_jobs)
            name_to_status = dict(update(list(stale_job_names)))
            for job in stale_jobs:
                status = name_to_status[job.name]
                if status:
                    # XXX: Why do we unpack that status? Can we just pass a
                    #      reference to the job?? Does the job not have a
                    #      reference to the "most up to date" status bc we
                    #      explicitly want copies??
                    job.set_status(
                        status.status,
                        status.launcher_status,
                        status.returncode,
                        error=status.error,
                        output=status.output,
                    )

    @classmethod
    def stop_all(cls, jobs: t.Iterable["Job"]) -> None:
        jobs = _helpers.unique(jobs)
        active = (job for job in jobs if job.status not in TERMINAL_STATUSES)
        to_kill = _helpers.group_by_map(cls._get_stop_callback, active)
        for kill, jobs_ in to_kill.items():
            for job in jobs_:
                status = kill(job.name)
                # XXX: Same question as above
                job.set_status(
                    status.status,
                    status.launcher_status,
                    status.returncode,
                    error=status.error,
                    output=status.output,
                )

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
        :type return_code: int|None
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
        self,
        new_job_name: _core_types.StepName,
        new_job_id: _core_types.JobIdType,
        is_task: bool,
    ) -> None:
        """Reset the job in order to be able to restart it.

        :param new_job_name: name of the new job step
        :type new_job_name: str
        :param new_job_id: new job id to launch under
        :type new_job_id: int
        :param is_task: process monitored by TaskManager (True) or the WLM (True)
        :type is_task: bool
        """
        # XXX: Very scary casts here
        self.name = new_job_name
        self.jid = new_job_id
        self.status = SmartSimStatus.STATUS_NEW
        self.returncode = None
        self.output = None
        self.error = None
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
        return f"{self.ename}{f'({self.jid})' if self.jid else ''}: {self.status}"


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


def _attempt_graceful_db_node_shutdown(
    dbnode: DBNode, fallback: t.Callable[["_core_types.StepName"], StepInfo]
) -> t.Callable[["_core_types.StepName"], StepInfo]:
    def _inner(step_name: "_core_types.StepName") -> StepInfo:
        ips = (network.get_ip_from_host(host) for host in dbnode.hosts)
        shutdown_results = (
            redis.shutdown_db_node(ip, port)
            for ip, port in itertools.product(ips, dbnode.ports)
        )
        return_codes = (ret for ret, _, _ in shutdown_results)
        if not all(ret == 0 for ret in return_codes):
            return fallback(step_name)
        return StepInfo(SmartSimStatus.STATUS_CANCELLED, "", 0, output=None, error=None)

    return _inner
