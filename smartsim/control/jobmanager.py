import time
import numpy as np
from threading import Thread

from .job import Job
from ..database import Orchestrator
from ..launcher.launcher import Launcher
from ..launcher.taskManager import Status
from ..entity import SmartSimEntity, Ensemble
from ..error import SmartSimError, SSConfigError

from ..utils import get_logger, get_env

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

    def __init__(self, launcher=None):
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

    def start(self):
        """Start a thread for the job manager"""
        monitor = Thread(name=self.name, daemon=True, target=self.run)
        monitor.start()

    def run(self):
        """Start the JobManager thread to continually check
        the status of all jobs.
        """
        # TODO a way to control this interval as a configuration
        interval = 5
        logger.debug("Starting Job Manager thread: " + self.name)

        self.actively_monitoring = True
        while self.actively_monitoring:
            time.sleep(interval)
            logger.debug(f"{self.name} - Active Jobs: {list(self().keys())}")

            # check each task
            for name, job in self().items():
                self.check_job(name)

                # if the job has errors then output the report
                # this should only output once
                if job.error:
                    logger.warning(job)
                    logger.warning(job.error_report())
                    self.move_to_completed(job)

                # Dont actively monitor completed jobs
                if self._launcher.is_finished(job.status):
                    logger.info(job)
                    self.move_to_completed(job)

            # if no more jobs left to actively monitor
            if not self():
                self.actively_monitoring = False
                logger.debug(f"{self.name} - Sleeping, no jobs to monitor")

    def move_to_completed(self, job):
        """Move job to completed queue so that its no longer
           actively monitored by the job manager

        :param job: job instance we are transitioning
        :type job: Job
        """
        self.completed[job.name] = job
        job.record_history()

        # remove from actively monitored jobs
        if job.name in self.db_jobs.keys():
            del self.db_jobs[job.name]
        elif job.name in self.jobs.keys():
            del self.jobs[job.name]

    def __getitem__(self, job_name):
        """Return the job associated with the job_name

        :param job_name: The name of the job
        :type job_name: str
        :returns: the Job associated with the job_name
        :rtype: Job
        """
        if job_name in self.db_jobs.keys():
            return self.db_jobs[job_name]
        elif job_name in self.jobs.keys():
            return self.jobs[job_name]
        elif job_name in self.completed.keys():
            return self.completed[job_name]
        else:
            raise KeyError

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
        job = self[name]
        if job.nodes:
            return job.nodes
        else:
            time.sleep(wait)
            nodes = self._launcher.get_step_nodes(job.jid)
            job.nodes = nodes
            return nodes

    def add_job(self, name, job_id, entity):
        """Add a job to the job manager which holds specific jobs by type.

        :param name: job name (usually the entity name)
        :type name: str
        :param job_id: job step id created by launcher
        :type job_id: str
        :param entity: entity that was launched on job step
        :type entity: SmartSimEntity
        """
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
        nodes = []
        for db_job in self.db_jobs.values():
            nodes.extend(db_job.nodes)
        return nodes

    def check_job(self, entity_name):
        """Update job properties by querying the launcher

        :param entity_name: name of the entity launched that
                            is to have it's status updated
        :type entity_name: str
        """
        try:
            if entity_name not in self.completed.keys():
                job = self[entity_name]
                status = self._launcher.get_step_status(job.jid)

                job.set_status(
                    status.status,
                    status.returncode,
                    error=status.error,
                    output=status.output,
                )
        except SmartSimError:
            logger.warning(f"Could not retrieve status of {entity_name}")

    def get_status(self, entity):
        """Return the workload manager given status of a job.

        :param entity: object launched by SmartSim. One of the following:
                    (SmartSimNode, Model, Orchestrator, Ensemble)
        :type entity: SmartSimEntity
        :returns: tuple of status
        """
        try:
            if entity.name in self.completed:
                return self.completed[entity.name].status

            job = self[entity.name]
            self.check_job(entity.name)
        except KeyError:
            raise SmartSimError(
                f"Entity by the name of {entity.name} has not been launched by this Controller"
            )
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
        job = self.completed[job_name]
        del self.completed[job_name]
        job.reset(new_job_id)
        if job.entity.type == "db":
            self.db_jobs[job_name] = job
        else:
            self.jobs[job_name] = job