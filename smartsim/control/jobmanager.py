import time
from .job import Job
from ..error import SmartSimError

from ..entity import SmartSimEntity, Ensemble
from ..orchestrator import Orchestrator
from ..launcher.launcher import Launcher

from ..utils import get_logger
logger = get_logger(__name__)


class JobManager:
    """The JobManager maintains a mapping between user defined entities
       and the steps launched through the workload manager. The JobManager
       holds jobs according to entity type.

       The JobManager and Controller share a single instance of a launcher
       object that allows both the Controller and launcher access to the
       wlm to query information about jobs that the user requests.
    """

    def __init__(self, launcher=None):
        self._launcher = launcher
        self.jobs = {}
        self.db_jobs = {}
        self.node_jobs = {}

    def __getitem__(self, job_name):
        if job_name in self.db_jobs.keys():
            return self.db_jobs[job_name]
        elif job_name in self.node_jobs.keys():
            return self.node_jobs[job_name]
        elif job_name in self.jobs.keys():
            return self.jobs[job_name]
        else:
            raise KeyError

    def __call__(self):
        all_jobs = {
            **self.jobs,
            **self.node_jobs,
            **self.db_jobs
            }
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
        """Add a job to the job manager which holds
           specific jobs by type.

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
        elif entity.type == "node":
            self.node_jobs[name] = job
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
            job = self[entity_name]
            status, returncode = self._launcher.get_step_status(job.jid)
            job.set_status(status, returncode)
        except SmartSimError:
            logger.warning(f"Could not retrieve status of {entity_name}")


    def get_status(self, entity):
        """Return the workload manager given status of a job.

           :param entity: object launched by SmartSim. One of the following:
                          (SmartSimNode, NumModel, Orchestrator, Ensemble)
           :type entity: SmartSimEntity
           :returns: tuple of status
        """
        try:
            job = self[entity.name]
            self.check_job(entity.name)
        except KeyError:
            raise SmartSimError(
                f"Entity by the name of {entity.name} has not been launched by this Controller")
        return job.status

    def poll(self, ignore_db, verbose):
        """Poll all simulations and return a boolean for
           if all jobs are finished or not.

           :param bool verbose: set verbosity
           :param bool ignore_db: return true even if the orchestrator nodes are still running
           :returns: True or False for if all models have finished
        """
        finished = True
        for job in self().values():
            if ignore_db and job.entity.type == "db":
                continue
            else:
                self.check_job(job.entity.name)
                if not self._launcher.is_finished(job.status):
                    finished = False
                if verbose:
                    logger.info(job)
        return finished

    def finished(self, entity):
        """Return a boolean indicating wether a job has finished or not

           :param entity: object launched by SmartSim. One of the following:
                          (SmartSimNode, NumModel, Ensemble)
           :type entity: SmartSimEntity
           :returns: bool
        """
        try:
            if isinstance(entity, Orchestrator):
                raise SmartSimError(
                    "Finished() does not support Orchestrator instances")
            if not isinstance(entity, SmartSimEntity):
                raise SmartSimError(
                    "Finished() only takes arguments of SmartSimEntity instances")
            if isinstance(entity, Ensemble):
                return all([self.finished(model) for model in entity.models.values()])

            job = self[entity.name]
            self.check_job(entity.name)
            return self._launcher.is_finished(job.status)
        except KeyError:
            raise SmartSimError(
                f"Entity by the name of {entity.name} has not been launched by this Controller")

    def _set_launcher(self, launcher):
        """Set the launcher of the job manager to a specific launcher
           instance created by the controller.

        :param launcher: child of Launcher
        :type launcher: Launcher instance
        """
        self._launcher = launcher

    def job_exists(self, entity):
        """Return a boolean for existence of a job in the jobmanager

        :param entity: Entity to check for job
        :type entity: SmartSimEntity
        :return: boolean indicating the existence of a job
        :rtype: bool
        """
        try:
            if self[entity.name]:
                return True
        except KeyError:
            return False
