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

import itertools
import time
import typing as t
from threading import Thread, RLock
from types import FrameType

from ...database import Orchestrator
from ...entity import DBNode, SmartSimEntity, EntityList
from ...error import SmartSimError
from ...log import get_logger
from ...status import TERMINAL_STATUSES
from ..config import CONFIG
from ..launcher import LocalLauncher, Launcher
from ..utils.network import get_ip_from_host
from .job import Job

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

    def __init__(self, lock: RLock, launcher: t.Optional[Launcher] = None) -> None:
        """Initialize a Jobmanager

        :param launcher: a Launcher object to manage jobs
        :type: SmartSim.Launcher
        """
        self.monitor: t.Optional[Thread] = None

        # active jobs
        self.jobs: t.Dict[str, Job] = {}
        self.db_jobs: t.Dict[str, Job] = {}

        # completed jobs
        self.completed: t.Dict[str, Job] = {}

        self.actively_monitoring = False  # on/off flag
        self._launcher = launcher  # reference to launcher
        self._lock = lock  # thread lock

        self.kill_on_interrupt = True  # flag for killing jobs on SIGINT

    def start(self) -> None:
        """Start a thread for the job manager"""
        self.monitor = Thread(name="JobManager", daemon=True, target=self.run)
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
            self.check_jobs()  # update all job statuses at once
            for _, job in self().items():
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
            if not self():
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

    def __getitem__(self, entity_name: str) -> Job:
        """Return the job associated with the name of the entity
        from which it was created.

        :param entity_name: The name of the entity of a job
        :type entity_name: str
        :returns: the Job associated with the entity_name
        :rtype: Job
        """
        with self._lock:
            if entity_name in self.db_jobs:
                return self.db_jobs[entity_name]
            if entity_name in self.jobs:
                return self.jobs[entity_name]
            if entity_name in self.completed:
                return self.completed[entity_name]
            raise KeyError

    def __call__(self) -> t.Dict[str, Job]:
        """Returns dictionary all jobs for () operator

        :returns: Dictionary of all jobs
        :rtype: dictionary
        """
        all_jobs = {**self.jobs, **self.db_jobs}
        return all_jobs

    def add_job(
        self,
        job_name: str,
        job_id: t.Optional[str],
        entity: t.Union[SmartSimEntity, EntityList],
        is_task: bool = True,
    ) -> None:
        """Add a job to the job manager which holds specific jobs by type.

        :param job_name: name of the job step
        :type job_name: str
        :param job_id: job step id created by launcher
        :type job_id: str
        :param entity: entity that was launched on job step
        :type entity: SmartSimEntity | EntityList
        :param is_task: process monitored by TaskManager (True) or the WLM (True)
        :type is_task: bool
        """
        launcher = str(self._launcher)
        # all operations here should be atomic
        job = Job(job_name, job_id, entity, launcher, is_task)
        if isinstance(entity, (DBNode, Orchestrator)):
            self.db_jobs[entity.name] = job
        else:
            self.jobs[entity.name] = job

    def is_finished(self, entity: SmartSimEntity) -> bool:
        """Detect if a job has completed

        :param entity: entity to check
        :type entity: SmartSimEntity
        :return: True if finished
        :rtype: bool
        """
        with self._lock:
            job = self[entity.name]  # locked operation
            if entity.name in self.completed:
                if job.status in TERMINAL_STATUSES:
                    return True
            return False

    def check_jobs(self) -> None:
        """Update all jobs in jobmanager

        Update all jobs returncode, status, error and output
        through one call to the launcher.

        """
        with self._lock:
            jobs = self().values()
            job_name_map = {job.name: job.ename for job in jobs}

            # returns (job step name, StepInfo) tuples
            if self._launcher:
                step_names = list(job_name_map.keys())
                statuses = self._launcher.get_step_update(step_names)
                for job_name, status in statuses:
                    job = self[job_name_map[job_name]]

                    if status:
                        # uses abstract step interface
                        job.set_status(
                            status.status,
                            status.launcher_status,
                            status.returncode,
                            error=status.error,
                            output=status.output,
                        )

    def get_status(self, entity: t.Union[SmartSimEntity, EntityList]) -> str:
        """Return the status of a job.

        :param entity: SmartSimEntity or EntityList instance
        :type entity: SmartSimEntity | EntityList
        :returns: tuple of status
        """
        with self._lock:
            try:
                if entity.name in self.completed:
                    return self.completed[entity.name].status

                job: Job = self[entity.name]  # locked
            except KeyError:
                raise SmartSimError(
                    f"Entity {entity.name} has not been launched in this Experiment"
                ) from None

            return job.status

    def set_launcher(self, launcher: Launcher) -> None:
        """Set the launcher of the job manager to a specific launcher instance

        :param launcher: child of Launcher
        :type launcher: Launcher instance
        """
        self._launcher = launcher

    def query_restart(self, entity_name: str) -> bool:
        """See if the job just started should be restarted or not.

        :param entity_name: name of entity to check for a job for
        :type entity_name: str
        :return: if job should be restarted instead of started
        :rtype: bool
        """
        if entity_name in self.completed:
            return True
        return False

    def restart_job(
        self,
        job_name: str,
        job_id: t.Optional[str],
        entity_name: str,
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

    def get_db_host_addresses(self) -> t.List[str]:
        """Retrieve the list of hosts for the database

        :return: list of host ip addresses
        :rtype: list[str]
        """
        addresses = []
        for db_job in self.db_jobs.values():
            if isinstance(db_job.entity, (DBNode, Orchestrator)):
                db_entity = db_job.entity

                for combine in itertools.product(db_job.hosts, db_entity.ports):
                    ip_addr = get_ip_from_host(combine[0])
                    addresses.append(":".join((ip_addr, str(combine[1]))))
        return addresses

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
                for dbnode in orchestrator.dbnodes:
                    if not dbnode.is_mpmd:
                        self.db_jobs[dbnode.name].hosts = [dbnode.host]
                    else:
                        self.db_jobs[dbnode.name].hosts = dbnode.hosts

    def signal_interrupt(self, signo: int, _frame: t.Optional[FrameType]) -> None:
        if not signo:
            logger.warning("Received SIGINT with no signal number")
        """Custom handler for whenever SIGINT is received"""
        if self.actively_monitoring and len(self) > 0:
            if self.kill_on_interrupt:
                for _, job in self().items():
                    if job.status not in TERMINAL_STATUSES and self._launcher:
                        self._launcher.stop(job.name)
            else:
                logger.warning("SmartSim process interrupted before resource cleanup")
                logger.warning("You may need to manually stop the following:")

                for job_name, job in self().items():
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
        local_jm_interval = 2
        if isinstance(self._launcher, (LocalLauncher)):
            time.sleep(local_jm_interval)
        else:
            time.sleep(CONFIG.jm_interval)

    def __len__(self) -> int:
        # number of active jobs
        return len(self.db_jobs) + len(self.jobs)
