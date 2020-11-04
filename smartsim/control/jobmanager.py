import time
import numpy as np
import threading
from threading import Thread

from .job import Job
from ..entity import DBNode
from ..database import Orchestrator
from ..launcher.launcher import Launcher
from ..launcher import SlurmLauncher, LocalLauncher
from ..entity import SmartSimEntity, Ensemble
from ..error import SmartSimError, SSConfigError
from ..utils import get_logger, get_env
from ..constants import LOCAL_JM_INTERVAL, WLM_JM_INTERVAL
from ..constants import TERMINAL_STATUSES

logger = get_logger(__name__)

class JobManager:
    """The JobManager maintains a mapping between user defined entities
    and the steps launched through the workload manager. The JobManager
    holds jobs according to entity type.

    The JobManager is threaded and runs during the course of an experiment
    to update the statuses of Jobs.

    The JobManager and Controller share a single instance of a launcher
    object that allows both the Controller and launcher access to the
    wlm to query information about jobs that the user requests.
    """

    def __init__(self, lock, launcher=None):
        """Initialize a Jobmanager

        :param launcher: a Launcher object to manage jobs
        :type: SmartSim.Launcher
        """
        self.name = "JobManager" + "-" + str(np.base_repr(time.time_ns(), 36))
        self.actively_monitoring = False
        self._launcher = launcher
        self.jobs = {}
        self.db_jobs = {}
        self.completed = {}
        self._lock = lock

    def start(self, daemon=True):
        """Start a thread for the job manager"""
        self.monitor = Thread(name=self.name, daemon=daemon, target=self.run)
        self.monitor.start()

    def run(self):
        """Start the JobManager thread to continually check
        the status of all jobs.
        """
        logger.debug("Starting Job Manager")

        self.actively_monitoring = True
        while self.actively_monitoring:
            self._thread_sleep()
            #logger.debug(f"Active Jobs: {len(self)}")

            # update all job statuses at once
            self.check_jobs()
            for name, job in self().items():

                # if the job has errors then output the report
                # this should only output once
                if job.returncode != None and job.status in TERMINAL_STATUSES:
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
                logger.debug(f"Sleeping, no jobs to monitor")

    def move_to_completed(self, job):
        """Move job to completed queue so that its no longer
           actively monitored by the job manager

        :param job: job instance we are transitioning
        :type job: Job
        """
        self._lock.acquire()
        try:
            self.completed[job.name] = job
            job.record_history()

            # remove from actively monitored jobs
            if job.name in self.db_jobs.keys():
                del self.db_jobs[job.name]
            elif job.name in self.jobs.keys():
                del self.jobs[job.name]
        finally:
            self._lock.release()

    def __getitem__(self, job_name):
        """Return the job associated with the job_name

        :param job_name: The name of the job
        :type job_name: str
        :returns: the Job associated with the job_name
        :rtype: Job
        """
        self._lock.acquire()
        try:
            if job_name in self.db_jobs.keys():
                return self.db_jobs[job_name]
            elif job_name in self.jobs.keys():
                return self.jobs[job_name]
            elif job_name in self.completed.keys():
                return self.completed[job_name]
            else:
                raise KeyError
        finally:
            self._lock.release()

    def __call__(self):
        """Returns dictionary all jobs for () operator

        :returns: Dictionary of all jobs
        :rtype: dictionary
        """
        all_jobs = {**self.jobs, **self.db_jobs}
        return all_jobs

    def get_job_nodes(self, name, wait=2):
        """Get the hostname(s) of a job from the allocation

        Wait time is necessary because Slurm take about 3 seconds to
        register that a job has been submitted.

        :param name: name of the launched entity
        :type name: str
        :param wait: time to wait before finding the hostname
                     aides in slow wlm launch times, defaults to 2
        :type wait: int, optional
        :return: hostnames of the job nodes where the entity was launched
        :rtype: list of str
        """
        self._lock.acquire()
        try:
            job = self[name]
            if job.nodes:
                return job.nodes
            else:
                time.sleep(wait)
                nodes = self._launcher.get_step_nodes(job.jid)
                job.nodes = nodes
                return nodes
        finally:
            self._lock.release()


    def add_job(self, name, job_id, entity):
        """Add a job to the job manager which holds specific jobs by type.

        :param name: job name (usually the entity name)
        :type name: str
        :param job_id: job step id created by launcher
        :type job_id: str
        :param entity: entity that was launched on job step
        :type entity: SmartSimEntity
        """
        # all operations here should be atomic
        job = Job(name, job_id, entity)
        if entity.type == "db":
            self.db_jobs[name] = job
        else:
            self.jobs[name] = job

    def get_db_hostnames(self):
        """Return a list of database nodes for cluster creation

        :return: list of hostnames that the database was
                 launched on.
        :rtype: list of strings
        """
        self._lock.acquire()
        try:
            nodes = []
            for db_job in self.db_jobs.values():
                nodes.extend(db_job.nodes)
            return nodes
        finally:
            self._lock.release()

    def is_finished(self, entity):
        self._lock.acquire()
        try:
            job = self[entity.name] # locked operation
            if entity.name in self.completed:
                if job.status in TERMINAL_STATUSES:
                    return True
            return False
        finally:
            self._lock.release()

    def check_jobs(self):
        """Update all jobs in jobmanager

        Update all jobs returncode, status, error and output
        through one call to the launcher.
        """
        self._lock.acquire()
        try:
            jobs = self().values()
            jids = [job.jid for job in jobs]
            statuses = self._launcher.get_step_update(jids)
            for status, job in zip(statuses, jobs):
                job.set_status(
                    status.status,
                    status.returncode,
                    error=status.error,
                    output=status.output
                )
        finally:
            self._lock.release()

    def get_status(self, entity):
        """Return the workload manager given status of a job.

        :param entity: object launched by SmartSim. One of the following:
                    (Model, DBNode)
        :type entity: SmartSimEntity
        :returns: tuple of status
        """
        self._lock.acquire()
        try:
            if entity.name in self.completed:
                return self.completed[entity.name].status

            job = self[entity.name] # locked
        except KeyError:
            raise SmartSimError(
                f"Entity by the name of {entity.name} has not been launched by this Controller"
            )
        finally:
            self._lock.release()
        return job.status

    def _set_launcher(self, launcher):
        """Set the launcher of the job manager to a specific launcher instance

        :param launcher: child of Launcher
        :type launcher: Launcher instance
        """
        self._launcher = launcher

    def query_restart(self, job_name):
        """See if the job just started should be restarted or not.

        :param job_name: name of the job
        :type job_name: str
        :return: if job should be restarted instead of started
        :rtype: bool
        """
        if job_name in self.completed:
            return True
        return False

    def restart_job(self, job_name, new_job_id):
        """Function to reset a job to record history and be
        ready to launch again.

        :param job_name: name of the job
        :type job_name: str
        :param new_job_id: new job id from the launcher
        :type new_job_id: str
        """
        self._lock.acquire()
        try:
            job = self.completed[job_name]
            del self.completed[job_name]
            job.reset(new_job_id)
            if job.entity.type == "db":
                self.db_jobs[job_name] = job
            else:
                self.jobs[job_name] = job
        finally:
            self._lock.release()

    def _thread_sleep(self):
        """Sleep the job manager for a specific constant
        set for the launcher type.
        """
        if isinstance(self._launcher, SlurmLauncher):
            time.sleep(WLM_JM_INTERVAL)
        else:
            time.sleep(LOCAL_JM_INTERVAL)

    def __len__(self):
        # number of active jobs
        return len(self.db_jobs) + len(self.jobs)
