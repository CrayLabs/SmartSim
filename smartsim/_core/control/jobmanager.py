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
import time
import typing as t
from collections import ChainMap
from threading import RLock, Thread
from types import FrameType

from ...database import Orchestrator
from ...entity import DBNode, EntitySequence, SmartSimEntity
from ...log import ContextThread, get_logger
from ...status import TERMINAL_STATUSES, SmartSimStatus
from ..config import CONFIG
from ..utils import helpers as _helpers
from ..launcher import Launcher, LocalLauncher
from ..utils.network import get_ip_from_host
from .job import Job, JobEntity

if t.TYPE_CHECKING:
    from smartsim._core import types as _core_types
    from smartsim.entity import types as _entity_types

logger = get_logger(__name__)


class JobManager:
    """The JobManager maintains a mapping between user defined entities
    and the steps launched through the launcher. The JobManager
    holds jobs according to entity type.

    The JobManager is threaded and runs during the course of an experiment
    to update the statuses of Jobs.

    The JobManager and Controller share a single instance of a launcher
    object that allows both the Controller and launcher access to the
    wlm to query information about jobs that the user requests.
    """

    def __init__(
        self,
        lock: RLock,
        launcher: t.Optional[Launcher] = None,
        poll_status_interval: int = CONFIG.jm_interval,
    ) -> None:
        """Initialize a Jobmanager

        :param launcher: a Launcher object to manage jobs
        :type: SmartSim.Launcher
        """
        self.monitor: t.Optional[Thread] = None

        # active jobs
        self.jobs: t.Dict["_entity_types.EntityName", Job] = {}
        self.db_jobs: t.Dict["_entity_types.EntityName", Job] = {}

        # completed jobs
        self.completed: t.Dict["_entity_types.EntityName", Job] = {}

        self.actively_monitoring = False  # on/off flag
        self._lock = lock  # thread lock

        self.kill_on_interrupt = True  # flag for killing jobs on SIGINT
        self._poll_status_interval = poll_status_interval

    def start(self) -> None:
        """Start a thread for the job manager"""
        self.monitor = ContextThread(name="JobManager", daemon=True, target=self.run)
        self.monitor.start()

    def run(self) -> None:
        """Start the JobManager thread to continually check
        the status of all jobs. Whichever launcher is selected
        by the user will be responsible for returning statuses
        that progress the state of the job.

        The interval of the checks is controlled by
        smartsim.constats.TM_INTERVAL and should be set to values
        above 20 for congested, multi-user systems

        The job manager thread will exit when no jobs are left
        or when the main thread dies
        """
        logger.debug("Starting Job Manager")
        self.actively_monitoring = True
        while self.actively_monitoring:
            self._thread_sleep()
            self.update_statuses()  # update all job statuses at once
            for job in self.get_active_jobs().values():
                # if the job has errors then output the report
                # this should only output once
                if job.returncode is not None and job.status in TERMINAL_STATUSES:
                    if int(job.returncode) != 0:
                        logger.warning(job)
                        logger.warning(job.error_report())
                        self.move_to_completed(job)
                    else:
                        # job completed without error
                        logger.info(job)
                        self.move_to_completed(job)

            # if no more jobs left to actively monitor
            if not self.get_active_jobs():
                self.actively_monitoring = False
                logger.debug("Sleeping, no jobs to monitor")

    def move_to_completed(self, job: Job) -> None:
        """Move job to completed queue so that its no longer
           actively monitored by the job manager

        :param job: job instance we are transitioning
        :type job: Job
        """
        with self._lock:
            self.completed[job.ename] = job
            job.record_history()

            # remove from actively monitored jobs
            if job.ename in self.db_jobs:
                del self.db_jobs[job.ename]
            elif job.ename in self.jobs:
                del self.jobs[job.ename]

    def __getitem__(self, entity_name: "_entity_types.EntityName") -> Job:
        """Return the job associated with the name of the entity
        from which it was created.

        :param entity_name: The name of the entity of a job
        :type entity_name: str
        :returns: the Job associated with the entity_name
        :rtype: Job
        """
        with self._lock:
            entities = ChainMap(self.db_jobs, self.jobs, self.completed)
            return entities[entity_name]

    def get_active_jobs(self) -> t.Mapping["_entity_types.EntityName", Job]:
        """Returns a mapping of entity name to job for all active jobs

        :returns: A mapping of entity name to job for all active jobs
        :rtype: Mapping[str, Job]
        """
        return ChainMap(self.db_jobs, self.jobs)

    def add_job(self, job: Job) -> None:
        """Add a job to the job manager which holds specific jobs by type.

        :param job_name: name of the job step
        :type job_name: str
        :param job_id: job step id created by launcher
        :type job_id: str
        :param entity: entity that was launched on job step
        :type entity: SmartSimEntity | EntitySequence
        :param is_task: process monitored by TaskManager (True) or the WLM (True)
        :type is_task: bool
        """
        # all operations here should be atomic
        if isinstance(job.entity, (DBNode, Orchestrator)):
            self.db_jobs[job.entity.name] = job
        elif isinstance(job.entity, JobEntity) and job.entity.is_db:
            self.db_jobs[job.entity.name] = job
        else:
            self.jobs[job.entity.name] = job

    def is_finished(self, entity: SmartSimEntity) -> bool:
        """Detect if a job has completed

        :param entity: entity to check
        :type entity: SmartSimEntity
        :return: True if finished
        :rtype: bool
        """
        with self._lock:
            job = self[entity.name]  # locked operation
            return entity.name in self.completed and job.status in TERMINAL_STATUSES

    def update_statuses(self) -> None:
        """Trigger a status of all monitored jobs

        Update all jobs returncode, status, error and output
        through one call to the launcher.
        """
        with self._lock:
            Job.refresh_all(self.get_active_jobs().values())

    def get_status(
        self,
        entity: t.Union[SmartSimEntity, EntitySequence[SmartSimEntity]],
    ) -> SmartSimStatus:
        """Return the status of a job.

        :param entity: SmartSimEntity or EntitySequence instance
        :type entity: SmartSimEntity | EntitySequence
        :returns: a SmartSimStatus status
        :rtype: SmartSimStatus
        """
        with self._lock:
            if entity.name in self.completed:
                return self.completed[entity.name].status

            try:
                return self[entity.name].status
            except KeyError:
                return SmartSimStatus.STATUS_NEVER_STARTED

    def find_completed_job(
        self, entity_name: "_entity_types.EntityName"
    ) -> t.Optional[Job]:
        """See if the job just started should be restarted or not.

        :param entity_name: name of entity to check for a job for
        :type entity_name: str
        :return: if job should be restarted instead of started
        :rtype: bool
        """
        return self.completed.get(entity_name, None)

    def restart_job(
    #   ^^^^^^^^^^^
    # XXX: Don't like this name, nothing is being "started", only tracked
        self,
        job_name: "_core_types.StepName",
        job_id: "_core_types.JobIdType",
        entity_name: "_entity_types.EntityName",
        is_task: bool = True,
    ) -> None:
        """Function to reset a job to record history and be
        ready to launch again.

        :param job_name: new job step name
        :type job_name: str
        :param job_id: new job id
        :type job_id: str
        :param entity_name: name of the entity of the job
        :type entity_name: str
        :param is_task: process monitored by TaskManager (True) or the WLM (True)
        :type is_task: bool
        """
        with self._lock:
            job = self.completed[entity_name]
            del self.completed[entity_name]
            job.reset(job_name, job_id, is_task)

            if isinstance(job.entity, (DBNode, Orchestrator)):
                self.db_jobs[entity_name] = job
            else:
                self.jobs[entity_name] = job

    def get_db_host_addresses(self) -> t.Dict[str, t.List[str]]:
        """Retrieve the list of hosts for the database
        for corresponding database identifiers

        :return: dictionary of host ip addresses
        :rtype: Dict[str, list]
        """

        address_dict: t.Dict[str, t.List[str]] = {}
        for db_job in self.db_jobs.values():
            if isinstance(db_job.entity, (DBNode, Orchestrator)):
                db_entity = db_job.entity

                dict_entry: t.List[str] = address_dict.get(db_entity.db_identifier, [])
                dict_entry.extend(
                    f"{get_ip_from_host(host)}:{port}"
                    for host, port in itertools.product(db_job.hosts, db_entity.ports)
                )
                address_dict[db_entity.db_identifier] = dict_entry

        return address_dict

    def set_db_hosts(self, orchestrator: Orchestrator) -> None:
        """Set the DB hosts in db_jobs so future entities can query this

        :param orchestrator: orchestrator instance
        :type orchestrator: Orchestrator
        """
        # should only be called during launch in the controller

        with self._lock:
            if orchestrator.batch:
                self.db_jobs[orchestrator.name].hosts = orchestrator.hosts

            else:
                for dbnode in orchestrator.entities:
                    if not dbnode.is_mpmd:
                        self.db_jobs[dbnode.name].hosts = [dbnode.host]
                    else:
                        self.db_jobs[dbnode.name].hosts = dbnode.hosts

    def signal_interrupt(self, signo: int, _frame: t.Optional[FrameType]) -> None:
        """Custom handler for whenever SIGINT is received"""
        if not signo:
            logger.warning("Received SIGINT with no signal number")
        if self.actively_monitoring and len(self.get_active_jobs()) > 0:
            if self.kill_on_interrupt:
                Job.stop_all(self.get_active_jobs().values())
            else:
                logger.warning("SmartSim process interrupted before resource cleanup")
                logger.warning("You may need to manually stop the following:")

                for job_name, job in self.get_active_jobs().items():
                    if job.is_task:
                        # this will be the process id
                        logger.warning(f"Task {job_name} with id: {job.jid}")
                    else:
                        logger.warning(
                            f"Job {job_name} with {job.launched_with} id: {job.jid}"
                        )

    def _thread_sleep(self) -> None:
        """Sleep the job manager for a specific constant
        set for the launcher type.
        """
        time.sleep(self._poll_status_interval)
