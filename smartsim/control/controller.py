import sys
import time
import subprocess
import pickle
from os import listdir
from os.path import isdir, basename, join

from ..launcher import SlurmLauncher, LocalLauncher
from ..orchestrator import Orchestrator
from ..entity import SmartSimEntity, DBNode, Ensemble, NumModel, SmartSimNode
from ..error import SmartSimError, SSConfigError, SSUnsupportedError, LauncherError
from ..launcher.clusterLauncher import create_cluster, check_cluster_status

from .job import Job
from .jobmanager import JobManager
from ..utils import get_config, remove_env

from ..utils import get_logger
logger = get_logger(__name__)


class Controller():
    """The controller module provides an interface between the
    numerical model that is the subject of Smartsim and the
    underlying workload manager or run framework.
    """

    def __init__(self, launcher="slurm"):
        """Initialize a Controller

        :param launcher: the type of launcher being used
        :type launcher: str
        """
        self._jobs = JobManager()
        self.init_launcher(launcher)


    def start(self, ensembles=None, nodes=None, orchestrator=None):
        """Start the Controller instance

        The function starts the computation of a ensemble, nodes, and
        optionally facilitate communication between entities via an
        orchestrator.  Controller.start() expects objects to be passed
        to the ensembles, nodes, and orchestrator objects. These are
        usually provided by Experiment when calling Experiment.start().

        :param ensembles: Ensembles to launch with specified launcher
        :type ensembles: a list of Ensemble objects
        :param nodes: SmartSimNodes to launch with specified launcher
        :type nodes: a list of SmartSimNode objects
        :param orchestrator: Orchestrator object to be launched for entity communication
        :type orchestrator: Orchestrator object
        """
        self._sanity_check_launch(ensembles, nodes, orchestrator)
        # TODO remove this, this is hacky
        models = []
        if ensembles:
            for ensemble in ensembles:
                models.extend(ensemble.get_models())
        self._launch(models, nodes, orchestrator)

        # start the job manager thread if not already started
        if not self._jobs.actively_monitoring:
            self._jobs.start()


    def stop(self, ensembles=None, models=None, nodes=None, orchestrator=None):
        """Stop the Controller instance

        Stops specified ensembles, nodes, and orchestrator.
        If stop_orchestrator is set to true and all ensembles and
        nodes are stopped, the orchestrator will be stopped.

        :param ensembles: List of ensembles to be stopped
        :type ensembles: list of ensemble, optional ensemble
        :param models: List of models to be stopped
        :type models: list of NumModel, option NumModel
        :param smartSimNode nodes: List of nodes to be stopped
        :type nodes: list of smartSimNode, optional smartSimNode
        :param bool stop_orchestrator: Boolean indicating if
                the orchestrator should be stopped.
        :raises: SSConfigError if called when using local launcher
        """

        if isinstance(ensembles, Ensemble):
            ensembles = [ensembles]
        if isinstance(nodes, SmartSimNode):
            nodes = [nodes]
        if isinstance(models, NumModel):
            models = [models]
        if orchestrator and not isinstance(orchestrator, Orchestrator):
            raise TypeError(
                f"Argument given for orchestrator is of type {type(orchestrator)}, not Orchestrator"
            )
        self._stop_ensembles(ensembles)
        self._stop_models(models)
        self._stop_nodes(nodes)
        if orchestrator:
            self._stop_orchestrator(orchestrator.dbnodes)

    def get_allocation(self, nodes=1, ppn=1, duration="1:00:00", **kwargs):
        """Get an allocation through the launcher

        This function gets an allocation through the launcher
        for future calls to start to launch entities onto.

        :param nodes: number of nodes for the allocation
        :type nodes: int
        :param ppn: processes per node
        :type ppn: int
        :param duration: length of the allocation in HH:MM:SS format,
                         defaults to "1:00:00"
        :type duration: str, optional
        :raises SmartSimError: if allocation could not be obtained
        :return: allocation id
        :rtype: str
        """
        logger.info("Attempting to obtain allocation...")
        try:
            alloc_id = self._launcher.get_alloc(
                nodes=nodes,
                ppn=ppn,
                duration=duration,
                **kwargs
            )
            return alloc_id
        except LauncherError as e:
            raise SmartSimError("Failed to get user requested allocation") from e

    def add_allocation(self, alloc_id):
        """Add an allocation to SmartSim for future entity launching

        :param alloc_id: id of the allocation from the workload manager
        :type alloc_id: str
        :raises SmartSimError: If the allocation cannot be found
        """
        try:
            alloc_id = str(alloc_id)
            self._launcher.accept_alloc(alloc_id)
            logger.info(f"Accepted user obtained allocation with id: {alloc_id}")
        except LauncherError as e:
            raise SmartSimError("Failed to accept user obtained allocation") from e

    def release(self, alloc_id=None):
        """Release the allocation(s)

        The function releases the allocation(s) stopping all
        jobs that are currently running and freeing up resources.
        If an allocation ID is provided, only stop that allocation
        and remove it from SmartSim.

        :param alloc_id: id of the allocation, defaults to None
        :type alloc_id: str, optional
        :raises SmartSimError: if fail to release allocation
        """
        try:
            if alloc_id:
                alloc_id = str(alloc_id)
                self._launcher.free_alloc(alloc_id)
            else:
                allocs = self._launcher.alloc_manager().copy()
                for alloc_id in allocs.keys():
                    self._launcher.free_alloc(alloc_id)
        except LauncherError as e:
            raise SmartSimError(f"Failed to release resources: {alloc_id}") from e

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
        all_finished = False
        while not all_finished:
            time.sleep(interval)
            finished = []
            for _, job in self._jobs().items():
                if not poll_db and job.entity.type == "db":
                    continue
                else:
                    finished.append(self._launcher.is_finished(job.status))
                    if verbose:
                        logger.info(job)
            if all(finished):
                all_finished = True


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

            job = self._jobs[entity.name]
            #self._jobs.check_job(entity.name) # shouldnt be necessary if jobmanager is running
            return self._launcher.is_finished(job.status)
        except KeyError:
            raise SmartSimError(
                f"Entity by the name of {entity.name} has not been launched by this Controller")


    def get_orchestrator_status(self, orchestrator):
        """Return the workload manager status of an Orchestrator

        This function returns the workload manager status of an
        orchestrator launched through the Controller.

        :param orchestrator: The Orchestrator instance to check the status of
        :type orchestrator: Orchestrator instance
        :returns: statuses of the orchestrator in a list
        :rtype: list of str
        """
        if not isinstance(orchestrator, Orchestrator):
            raise TypeError(
                f"orchestrator argument was of type {type(orchestrator)} not of type Orchestrator")
        statuses = []
        for dbnode in orchestrator.dbnodes:
            statuses.append(self._jobs.get_status(dbnode))
        return statuses

    def get_ensemble_status(self, ensemble):
        """Return the workload manager status of an ensemble of models

        This function returns the workload manager status
        of an ensemble of models launched through the Controller.

        :param ensemble: The Ensemble instance to check the status of
        :type ensemble: Ensemble instance
        :returns: statuses of the ensemble in a list
        :rtype: list of str
        """
        if not isinstance(ensemble, Ensemble):
            raise TypeError(
                f"ensemble argument was of type {type(ensemble)} not of type Ensemble")
        statuses = []
        for model in ensemble.models.values():
            statuses.append(self._jobs.get_status(model))
        return statuses

    def get_model_status(self, model):
        """Return the workload manager status of a model.

        :param model: the model to check the status of
        :type model: NumModel
        :returns: status of the model given by the workload manager
        :rtype: str
        """
        if not isinstance(model, NumModel):
            raise TypeError(
                f"model argument was of type {type(model)} not of type NumModel"
            )
        return self._jobs.get_status(model)

    def get_node_status(self, node):
        """Return the workload manager status of a SmartSimNode.

        :param node: the SmartSimNode to check the status of
        :type model: SmartSimNode
        :returns: status of the SmartSimNode given by the workload manager
        :rtype: str
        """
        if not isinstance(node, SmartSimNode):
            raise TypeError(
                f"node argument was of type {type(node)} not of type SmartSimNode"
            )
        return self._jobs.get_status(node)

    def init_launcher(self, launcher):
        """Initialize the controller with a specific type of launcher.

        Remove SMARTSIM_REMOTE env var if set as we are creating
        a new launcher that should not be effected by previous
        launcher settings in the environment

        Since the JobManager and the controller share a launcher
        instance, set the JobManager launcher if we create a new
        launcher instance here.

        :param launcher: which launcher to initialize
        :type launcher: str
        :raises SSUnsupportedError: if a string is passed that is not
                                    a supported slurm launcher
        :raises SSConfigError: if no launcher argument is provided.
        """
        remove_env("SMARTSIM_REMOTE")

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
                raise SSUnsupportedError("Launcher type not supported: " +
                                         launcher)
        else:
            raise SSConfigError(
                "Must provide a 'launcher' argument")

    def _stop_ensembles(self, ensembles):
        """Stops specified ensembles.

        This function stops the specified ensembles.
        If ensembles is None, the function returns without
        performing any action.

        :param ensembles: List of ensembles to be stopped
        :type ensembles: list of Ensemble instances
        :raises TypeError: if provided models is not SmartSim Ensemble
        """
        if not ensembles:
            return

        if not all(isinstance(x, Ensemble) for x in ensembles):
            raise TypeError(
                "Only objects of type Ensemble expected for input variable ensembles"
            )

        for ensemble in ensembles:
            models = list(ensemble.models.values())
            self._stop_models(models)

    def _stop_models(self, models):
        """Stops the specified models.

        This function stops the specified models.  If
        the models is None, the function returns
        without performing any action.

        :param models: List of models to be stopped
        :type models: list of NumModel instances
        :raises TypeError: if provided models is not SmartSim Model
        """

        if not models:
            return

        if not all(isinstance(x, NumModel) for x in models):
            raise TypeError(
                "Only objects of type NumModel expected for input variable models")

        for model in models:
            job = self._jobs[model.name]
            logger.info(" ".join((
                "Stopping model", model.name, "job", str(job.jid))))
            status = self._launcher.stop(job.jid)
            job.set_status(status.status,
                           status.returncode,
                           error=status.error,
                           output=status.output)
            self._jobs.move_to_completed(job)

    def _stop_nodes(self, nodes):
        """Stops specified nodes.

        If nodes is None, the function returns without performing
        any action.

        :param nodes: List of nodes to be stopped
        :type nodes: list of SmartSimNode instances
        :raises TypeError: if provided models is not SmartSimNode
        """
        if not nodes:
            return

        if not all(isinstance(x, SmartSimNode) for x in nodes):
            raise TypeError(
                "Only objects of type SmartSimNode expected for input variable nodes"
            )

        for node in nodes:
            job = self._jobs[node.name]
            logger.info("Stopping node " + node.name + " job " +
                        job.jid)
            status = self._launcher.stop(job.jid)
            job.set_status(status.status,
                           status.returncode,
                           error=status.error,
                           output=status.output)
            self._jobs.move_to_completed(job)

    def _stop_orchestrator(self, dbnodes):
        """Stops the orchestrator jobs that are currently running.

        :param dbnodes: the databases that make up the orchestrator
        :type dbnodes: a list of DBNode instances
        :raises TypeError: if provided db_nodes are not DBNode
        """
        if not dbnodes:
            return

        if not all(isinstance(x, DBNode) for x in dbnodes):
            raise TypeError(
                "Only objects of type DBNode expected for input variable dbnodes"
            )

        for dbnode in dbnodes:
            job = self._jobs[dbnode.name]
            logger.debug("Stopping orchestrator on job " + job.jid)
            status = self._launcher.stop(job.jid)
            job.set_status(status.status,
                           status.returncode,
                           error=status.error,
                           output=status.output)
            self._jobs.move_to_completed(job)


    def _launch(self, models, nodes, orchestrator):
        """Launch provided entities

        The main launching function of the Controller.
        Launches the orchestrator first in order to collect
        information about job placement to pass to nodes and
        models that need to connect to it. Each entity has
        a step created for it, and a list of those steps
        coupled with the entity objects in tuples are passed
        to each launch function for a dedicated entity type.

        :param models: list of all models passed to start
        :type models: list of NumModel instances
        :param nodes: list of all nodes passed to start
        :type nodes: list of SmartSimNode instances
        :param orchestrator: orchestrator instance
        :type orchestrator: Orchestrator
        """

        if orchestrator:
            active_orc = False
            if not isinstance(self._launcher, LocalLauncher):
                db_node = orchestrator.dbnodes[0]
                if db_node.name in self._jobs.db_jobs:
                    if not self.finished(db_node):
                        active_orc = True

            if not active_orc:
                for dbnode in orchestrator.dbnodes:
                    dbnode.remove_stale_dbnode_files()
                orc_steps = self._create_steps(orchestrator.dbnodes,
                                               orchestrator)

                self._launch_orchestrator(orc_steps, orchestrator)
                if len(orchestrator.dbnodes) > 2:
                    self._create_orchestrator_cluster(orchestrator)
                self._save_orchestrator(orchestrator)
                logger.debug(
                    f"Orchestrator launched on nodes: {self._jobs.get_db_hostnames()}")

        # orchestrator passed in to get env_vars
        node_steps = self._create_steps(nodes, orchestrator)
        model_steps = self._create_steps(models, orchestrator)
        self._launch_nodes(node_steps)
        self._launch_models(model_steps)


    def _create_steps(self, entities, orchestrator):
        """Create job steps for all entities with the launcher

        :param entities: list of all entities to create steps for
        :type entities: list of SmartSimEntities
        :param orchestrator: orchestrator instance
        :type orchestrator: Orchestrator
        :return: list of tuples of (launcher_step, entity)
        :rtype: list of tuples
        """
        steps = []
        if entities:
            steps = [(self._create_entity_step(entity, orchestrator), entity)
            for entity in entities]
        return steps

    def _create_entity_step(self, entity, orchestrator):
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
        :param orchestrator: orchestrator instance
        :type orchestrator: Orchestrator
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

            self._set_entity_env_vars(entity, orchestrator)
            step = self._launcher.create_step(entity.name,
                                              entity.run_settings,
                                              multi_prog=multi_prog)
            return step
        except LauncherError as e:
            error = f"Failed to create job step for {entity.name}"
            raise SmartSimError("\n".join((error, e.msg))) from None

    def _set_entity_env_vars(self, entity, orchestrator):
        """Set connection environment variables

        Retrieve the connections registered by the user for
        each entity and utilize the orchestrator for turning
        those connections into environment variables to launch
        with the step.

        :param entity: entity to find connections for
        :type entity: SmartSimEntity
        :param orchestrator: orchestrator instance
        :type orchestrator: Orchestrator
        """
        if orchestrator and not isinstance(entity, DBNode):
            env_vars = orchestrator.get_connection_env_vars(entity)
            existing_env_vars = entity.get_run_setting("env_vars")
            final_env_vars = {"env_vars": env_vars}
            if existing_env_vars:
                existing_env_vars.update(env_vars)
                final_env_vars["env_vars"] = existing_env_vars
            entity.update_run_settings(final_env_vars)

    def _launch_orchestrator(self, orc_steps, orchestrator):
        """Launch the orchestrator

        Launch the orchestrator instance as specified by the user.
        Immediately get the hostnames of the nodes where the
        orchestrator job was placed.

        :param orc_steps: Tuples of (step, entity) in a list
        :type orc_steps: list of tuples
        :param orchestrator: Orchestrator instance
        :type orchestrator: Orchestrator
        :raises SmartSimError: if orchestrator launch fails
        """
        for step_tuple in orc_steps:
            step, dbnode = step_tuple
            try:
                job_id = self._launcher.run(step)
                if self._jobs.query_restart(dbnode.name):
                    logger.debug(f"Restarting database node: {dbnode.name}")
                    self._jobs.restart_job(dbnode.name, job_id)
                else:
                    logger.debug(f"Launching database node: {dbnode.name}")
                    self._jobs.add_job(dbnode.name, job_id, dbnode)

                job_nodes = self._jobs.get_job_nodes(dbnode.name)
                orchestrator.junction.store_db_addr(job_nodes,
                                                    dbnode.ports)
            except LauncherError as e:
                logger.error(
                    "An error occured when launching the database nodes\n" +
                    "Check database node output files for details.")
                raise SmartSimError(
                    f"Database node {dbnode.name} failed to launch"
                    ) from None

    def _launch_nodes(self, node_steps):
        """Launch the SmartSimNode steps

        Launch all SmartSimNode steps using the launcher.
        Add the job to the job manager to keep track of the
        job information for the user.

        :param node_steps: steps for created_nodes
        :type node_steps: step object dependant on launcher
        :raises SmartSimError: if launch fails
        """
        for step_tuple in node_steps:
            step, node = step_tuple
            try:

                job_id = self._launcher.run(step)
                if self._jobs.query_restart(node.name):
                    logger.debug(f"Restarting SmartSimNode: {node.name}")
                    self._jobs.restart_job(node.name, job_id)
                else:
                    logger.debug(f"Launching SmartSimNode: {node.name}")
                    self._jobs.add_job(node.name, job_id, node)
            except LauncherError as e:
                logger.error(
                    "An error occured when launching SmartSimNodes\n" +
                    "Check node output files for details.")
                raise SmartSimError(
                    "SmartSimNode %s failed to launch" % node.name
                    ) from None

    def _launch_models(self, model_steps):
        """Launch the SmartSim models

        Launch all model steps using the launcher.
        Add the job to the job manager to keep track of the
        job information for the user.

        :param model_steps: tuples of (step, model)
        :type model_steps: list of tuples
        :raises SmartSimError: if any model fails to launch
        """
        for step_tuple in model_steps:
            step, model = step_tuple
            try:
                job_id = self._launcher.run(step)
                if self._jobs.query_restart(model.name):
                    logger.debug(f"Restarting Model: {model.name}")
                    self._jobs.restart_job(model.name, job_id)
                else:
                    logger.debug(f"Launching model: {model.name}")
                    self._jobs.add_job(model.name, job_id, model)
            except LauncherError as e:
                logger.error(
                    "An error occured when launching model ensembles.\n" +
                    "Check model output files for details.")
                raise SmartSimError(
                    "Model %s failed to launch" % model.name
                    ) from e


    def _create_orchestrator_cluster(self, orchestrator):
        """Create an orchestrator cluster

        If the number of database nodes is greater than 2
        we create a clustered orchestrator using this function.

        :param orchestrator: orchestrator instance
        :type orchestrator: Orchestrator
        """
        logger.debug("Constructing Orchestrator cluster...")
        all_ports = orchestrator.dbnodes[0].ports
        db_nodes = self._jobs.get_db_hostnames()
        create_cluster(db_nodes, all_ports)
        check_cluster_status(db_nodes, all_ports)

    def _sanity_check_launch(self, ensembles, nodes, orchestrator):
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
            if len(orchestrator.dbnodes) > 1:
                raise SSConfigError(
                    "Local launcher does not support launching multiple databases")

    def _save_orchestrator(self, orchestrator):
        """Save the orchestrator object via pickle

        This function saves the orchestrator information to a pickle
        file that can be imported by subsequent experiments to reconnect
        to the orchestrator.

        :param orchestrator: Orchestrator configuration to be saved
        :type orchestrator: Orchestrator
        """

        dat_file = "/".join((orchestrator.path,"smartsim_db.dat"))
        orc_data = {"orc":orchestrator, "db_jobs":self._jobs.db_jobs}
        with open(dat_file, "wb") as pickle_file:
            pickle.dump(orc_data, pickle_file)
