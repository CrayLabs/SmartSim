import sys
import time
import subprocess
import threading
import pickle
from os import listdir
from os.path import isdir, basename, join

from ..constants import TERMINAL_STATUSES
from ..database import Orchestrator
from ..launcher import SlurmLauncher, LocalLauncher
from ..entity import SmartSimEntity, DBNode, Ensemble, Model, EntityList
from ..launcher.clusterLauncher import create_cluster, check_cluster_status
from ..error import SmartSimError, SSConfigError, SSUnsupportedError, LauncherError

from .job import Job
from .junction import Junction
from .jobmanager import JobManager
from ..utils.entityutils import seperate_entities

from ..utils import get_logger

logger = get_logger(__name__)


class Controller:
    """The controller module provides an interface between the
    smartsim entities created in the experiment and the
    underlying workload manager or run framework.
    """

    def __init__(self, launcher="slurm"):
        """Initialize a Controller

        :param launcher: the type of launcher being used
        :type launcher: str
        """
        self._jobs = JobManager()
        self._cons = Junction()
        self.init_launcher(launcher)

    def start(self, *args, block=True):
        """Start the passed SmartSim entities

        This function should not be called directly, but rather
        through the experiment interface.

        The controller will start the job-manager thread upon
        execution of all jobs.
        """
        entities, entity_lists, orchestrator = seperate_entities(args)
        self._sanity_check_launch(orchestrator)

        self._launch(entities, entity_lists, orchestrator)

        # start the job manager thread if not already started
        if not self._jobs.actively_monitoring:
            self._jobs.start()

        # block until all non-database jobs are complete
        if block:
            self.poll(5, False, True)

    def poll(self, interval, poll_db, verbose):
        """Poll running simulations and receive logging output of job status

        :param interval: number of seconds to wait before polling again
        :type interval: int
        :param poll_db: poll dbnodes for status as well and see
                            it in the logging output
        :type poll_db: bool
        :param verbose: set verbosity
        :type verbose: bool
        """
        # fine to not lock here, nothing about the jobs
        # are being mutated in any way.
        to_monitor = self._jobs.jobs
        if poll_db:
            to_monitor = self._jobs()
            msg = "Monitoring database will loop infinitely as long"
            msg += "as database is alive.\n Ctrl+C to stop execution."
            logger.warning(msg)
        while len(to_monitor) > 0:
            time.sleep(interval)
            for job in to_monitor.values():
                if verbose:
                    logger.info(job)

    def finished(self, entity):
        """Return a boolean indicating wether a job has finished or not

        :param entity: object launched by SmartSim. One of the following:
                          (Entity, EntityList)
        :type entity: SmartSimEntity
        :returns: bool
        """
        try:
            if isinstance(entity, Orchestrator):
                raise SmartSimError(
                    "Finished() does not support Orchestrator instances"
                )
            if isinstance(entity, EntityList):
                return all([self.finished(ent) for ent in entity.entities])
            if not isinstance(entity, SmartSimEntity):
                raise SmartSimError(
                    f"Argument was of type {type(entity)} not SmartSimEntity or EntityList"
                )

            return self._jobs.is_finished(entity)
        except KeyError:
            raise SmartSimError(
                f"Entity by the name of {entity.name} has not been launched by this Controller"
            )

    def stop_entity(self, entity):
        """Stop an instance of an entity

        This function will also update the status of the job in
        the jobmanager so that the job appears as "cancelled".

        :param entity: entity to be stopped
        :type entity: SmartSimEntity
        """
        job = self._jobs[entity.name]
        if job.status not in TERMINAL_STATUSES:
            logger.info(" ".join(("Stopping model", entity.name, "job", str(job.jid))))
            status = self._launcher.stop(job.jid)
            job.set_status(
                status.status, status.returncode, error=status.error, output=status.output
            )
            self._jobs.move_to_completed(job)

    def stop_entity_list(self, entity_list):
        """Stop an instance of an entity list

        :param entity_list: entity list to be stopped
        :type entity_list: EntityList
        """
        for entity in entity_list.entities:
            self.stop_entity(entity)

    def get_entity_status(self, entity):
        """Get the status of an entity

        :param entity: entity to get status of
        :type entity: SmartSimEntity
        :raises TypeError: if not SmartSimEntity
        :return: status of entity
        :rtype: str
        """
        if not isinstance(entity, SmartSimEntity):
            raise TypeError(
                f"Argument was of type {type(entity)} not of type SmartSimEntity"
            )
        return self._jobs.get_status(entity)

    def get_entity_list_status(self, entity_list):
        """Get the statuses of an entity list

        :param entity_list: entity list containing entities to
                            get statuses of
        :type entity_list: EntityList
        :raises TypeError: if not EntityList
        :return: list of str statuses
        :rtype: list
        """
        if not isinstance(entity_list, EntityList):
            raise TypeError(f"Argument was of type {type(entity_list)} not EntityList")
        statuses = []
        for entity in entity_list.entities:
            statuses.append(self.get_entity_status(entity))
        return statuses

    def init_launcher(self, launcher):
        """Initialize the controller with a specific type of launcher.

        Since the JobManager and the controller share a launcher
        instance, set the JobManager launcher if we create a new
        launcher instance here.

        :param launcher: which launcher to initialize
        :type launcher: str
        :raises SSUnsupportedError: if a string is passed that is not
                                    a supported slurm launcher
        :raises SSConfigError: if no launcher argument is provided.
        """

        if launcher is not None:
            # Init Slurm Launcher wrapper
            if launcher == "slurm":
                self._launcher = SlurmLauncher()
                self._jobs._set_launcher(self._launcher)
            # Run all ensembles locally
            elif launcher == "local":
                self._launcher = LocalLauncher()
                self._jobs._set_launcher(self._launcher)
            else:
                raise SSUnsupportedError("Launcher type not supported: " + launcher)
        else:
            raise SSConfigError("Must provide a 'launcher' argument")

    def _launch(self, entities, entity_lists, orchestrator):
        """Main launching function of the controller

        Orchestrators are always launched first so that the
        address of the database can be given to following entities

        :param entities: entities to launch
        :type entities: SmartSimEntity
        :param entity_lists: entity lists to launch
        :type entity_lists: EntityList
        :param orchestrator: orchestrator to launch
        :type orchestrator: Orchestrator
        """
        if orchestrator and not self._active_orc(orchestrator):
            self._launch_orchestrator(orchestrator)

        # create all steps prior to launch
        steps = []
        if entity_lists:
            for elist in entity_lists:
                steps.extend(self._create_steps(elist.entities))
        if entities:
            steps.extend(self._create_steps(entities))

        # launch steps
        for step in steps:
            self._launch_entity(step)

    def _launch_orchestrator(self, orchestrator):
        """Launch an Orchestrator instance

        :param orchestrator: orchestrator to launch
        :type orchestrator: Orchestrator
        """
        orchestrator.remove_stale_files()
        database_steps = self._create_steps(orchestrator.entities)

        for orc_step in database_steps:
            self._launch_entity(orc_step)

        # create the database cluster
        if len(orchestrator.entities) > 2:
            self._create_orchestrator_cluster(orchestrator)
        self._save_orchestrator(orchestrator)
        logger.debug(f"Orchestrator launched on nodes: {self._jobs.get_db_hostnames()}")

    def _launch_entity(self, entity_step):
        """Launch an entity using initialized launcher

        :param entity_step: Step object dependant on launcher
        :type entity_step: Step object
        :raises SmartSimError: If launching fails
        """

        step, entity = entity_step
        try:
            job_id = self._launcher.run(step)
            if self._jobs.query_restart(entity.name):
                logger.debug(f"Restarting {entity.name}")
                self._jobs.restart_job(entity.name, job_id)
            else:
                logger.debug(f"Launching {entity.name}")
                self._jobs.add_job(entity.name, job_id, entity)

            if isinstance(entity, DBNode):
                job_nodes = self._jobs.get_job_nodes(entity.name)
                self._cons.store_db_addr(job_nodes, entity.ports)

        except LauncherError as e:
            logger.error(
                f"An error occurred when launching {entity} \n"
                + "Check error and output files for details."
            )
            raise SmartSimError(f"Job step {entity.name} failed to launch") from e

    def _create_steps(self, entities):
        """Create job steps for all entities with the launcher

        :param entities: list of all entities to create steps for
        :type entities: list of SmartSimEntities
        :return: list of tuples of (launcher_step, entity)
        :rtype: list of tuples
        """
        steps = []
        if entities:
            steps = [(self._create_entity_step(entity), entity) for entity in entities]
        return steps

    def _create_entity_step(self, entity):
        """Create a step for a single entity

         Steps are defined on the launcher level and are abstracted
         away from the Controller. Steps determine exactly how the
         entity is to be run and provide the input to Launcher.run().

        Optionally create a step that utilizes multiple programs
        within a single step as some workload managers like
        Slurm allow for. Currently this is only supported for
        launching multiple databases per node.

        :param entity: entity to create step for
        :type entity: SmartSimEntity
        :raises SmartSimError: if job step creation failed
        :return: the created job step
        :rtype: Step object (e.g. for Slurm its a SlurmStep)
        """
        multi_prog = False
        try:
            # launch in MPMD mode if database per node > 1
            if isinstance(entity, DBNode):
                if len(entity.ports) > 1:
                    multi_prog = True

            self._set_entity_env_vars(entity)
            step = self._launcher.create_step(
                entity.name, entity.run_settings, multi_prog=multi_prog
            )
            return step
        except LauncherError as e:
            error = f"Failed to create job step for {entity.name}"
            raise SmartSimError("\n".join((error, e.msg))) from None

    def _set_entity_env_vars(self, entity):
        """Set connection environment variables

        Retrieve the connections registered by the user for
        each entity and utilize the junction for turning
        those connections into environment variables to launch
        with the step.

        :param entity: entity to find connections for
        :type entity: SmartSimEntity
        """
        if not isinstance(entity, DBNode):
            env_vars = self._cons.get_connections(entity)
            existing_env_vars = entity.get_run_setting("env_vars")
            final_env_vars = {"env_vars": env_vars}
            if existing_env_vars:
                existing_env_vars.update(env_vars)
                final_env_vars["env_vars"] = existing_env_vars
            entity.update_run_settings(final_env_vars)

    def _create_orchestrator_cluster(self, orchestrator):
        """Create an orchestrator cluster

        If the number of database nodes is greater than 2
        we create a clustered orchestrator using this function.

        :param orchestrator: orchestrator instance
        :type orchestrator: Orchestrator
        """
        logger.debug("Constructing Orchestrator cluster...")
        all_ports = orchestrator.entities[0].ports
        db_nodes = self._jobs.get_db_hostnames()
        create_cluster(db_nodes, all_ports)
        check_cluster_status(db_nodes, all_ports)

    def _sanity_check_launch(self, orchestrator):
        """Check the orchestrator settings

        Sanity check the orchestrator settings in case the
        user tries to do something silly. This function will
        serve as the location of many such sanity checks to come.

        :param orchestrator: Orchestrator instance
        :type orchestrator: Orchestrator
        :raises SSConfigError: If local launcher is being used to
                               launch a database cluster
        """

        if isinstance(self._launcher, LocalLauncher) and orchestrator:
            if len(orchestrator) > 1:
                raise SSConfigError(
                    "Local launcher does not support launching multiple databases"
                )

    def _active_orc(self, orchestrator):
        """Detect if orchestrator is running already

        :param orchestrator: Orchestrator instance
        :type orchestrator: Orchestrator
        :return: if orchestrator is running or not
        :rtype: bool
        """
        active_orc = False
        if not isinstance(self._launcher, LocalLauncher):
            db_node = orchestrator.entities[0]
            if db_node.name in self._jobs.db_jobs:
                if not self.finished(db_node):
                    active_orc = True
        return active_orc

    def _save_orchestrator(self, orchestrator):
        """Save the orchestrator object via pickle

        This function saves the orchestrator information to a pickle
        file that can be imported by subsequent experiments to reconnect
        to the orchestrator.

        :param orchestrator: Orchestrator configuration to be saved
        :type orchestrator: Orchestrator
        """

        dat_file = "/".join((orchestrator.path, "smartsim_db.dat"))
        orc_data = {"orc": orchestrator, "db_jobs": self._jobs.db_jobs}
        with open(dat_file, "wb") as pickle_file:
            pickle.dump(orc_data, pickle_file)
