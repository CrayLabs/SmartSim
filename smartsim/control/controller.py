import sys
import time
import subprocess
from os import listdir
from os.path import isdir, basename, join

from ..ensemble import Ensemble
from ..model import NumModel
from ..launcher import SlurmLauncher, LocalLauncher
from ..smartSimNode import SmartSimNode
from ..orchestrator import Orchestrator
from ..dbnode import DBNode
from ..entity import SmartSimEntity
from ..error import SmartSimError, SSConfigError, SSUnsupportedError, LauncherError

from .job import Job
from .allochandler import AllocHandler
from .jobmanager import JobManager
from .dbcluster import create_cluster, check_cluster_status, kill_db_node
from ..utils import get_config

from ..utils import get_logger
logger = get_logger(__name__)


class Controller():
    """The controller module provides an interface between the numerical model
       that is the subject of Smartsim and the underlying workload manager or
       run framework.
    """
    def __init__(self):
        self._alloc_handler = AllocHandler()
        self._jobs = JobManager()
        self._launcher = None

    def start(self, ensembles=None, nodes=None, orchestrator=None, duration="1:00:00"):
        """Start the computation of a ensemble, nodes, and optionally
           facilitate communication between entities via an orchestrator.
           Controller.start() expects objects to be passed to the ensembles,
           nodes, and orchestrator objects. These are usually provided by
           Experiment when calling Experiment.start().

           :param ensembles: Ensembles to launch with specified launcher
           :type ensembles: a list of Ensemble objects
           :param nodes: SmartSimNodes to launch with specified launcher
           :type nodes: a list of SmartSimNode objects
           :param orchestrator: Orchestrator object to be launched for entity communication
           :type orchestrator: Orchestrator object
        """
        if isinstance(ensembles, Ensemble):
            ensembles = [ensembles]
        if isinstance(nodes, SmartSimNode):
            nodes = [nodes]
        if orchestrator and not isinstance(orchestrator, Orchestrator):
            raise SmartSimError(
                f"Argument given for orchestrator is of type {type(orchestrator)}, not Orchestrator"
            )
        self._prep_orchestrator(orchestrator)
        self._prep_nodes(nodes)
        self._prep_ensembles(ensembles)
        self._get_allocations(duration)
        self._launch(ensembles, nodes, orchestrator)


    def stop(self, ensembles=None, models=None, nodes=None, orchestrator=None):
        """Stops specified ensembles, nodes, and orchestrator.
           If stop_orchestrator is set to true and all ensembles and
           nodes are stopped, the orchestrator will be stopped.

           :param ensembles: List of ensembles to be stopped
           :type ensembles: list of ensemble, optional ensemble
           :param models: List of models to be stopped
           :type models: list of NumModel, option NumModel
           :param smartSimNode nodes: List of nodes to be stopped
           :type nodes: list of smartSimNode, optional smartSimNode
           :param bool stop_orchestrator: Boolean indicating if
                the ochestrator should be stopped.
            :raises: SSConfigError if called when using local launcher
        """
        self.check_local("Controller.stop()")

        if isinstance(ensembles, Ensemble):
            ensembles = [ensembles]
        if isinstance(nodes, SmartSimNode):
            nodes = [nodes]
        if isinstance(models, NumModel):
            models = [models]
        if orchestrator and not isinstance(orchestrator, Orchestrator):
            raise SmartSimError(
                f"Argument given for orchestrator is of type {type(orchestrator)}, not Orchestrator"
            )
        self._stop_ensembles(ensembles)
        self._stop_models(models)
        self._stop_nodes(nodes)
        if orchestrator:
            self._stop_orchestrator(orchestrator.dbnodes)

    def release(self):
        """Release the allocation(s) stopping all jobs that are currently running
           and freeing up resources.

           :raises: SSConfigError if called when using local launcher
           :raises: SmartSimError if partition could not be found
        """
        self.check_local("Controller.release()")

        allocs = self._alloc_handler.allocs.copy()
        for alloc_id in allocs.values():
            logger.info(f"Releasing allocation: {alloc_id}")

        for partition, alloc_id in allocs.items():
            self._launcher.free_alloc(alloc_id)
            self._alloc_handler._remove_alloc(partition)


    def get_job(self, name):
        """Retrieve a Job instance by name. The Job object carries information about the
           job launched by the controller and it's current status.

           :param str name: name of the entity launched by the Controller
           :raises: SmartSimError
        """
        try:
            return self._jobs[name]
        except KeyError:
            raise SmartSimError("Job for " + name + " not found.")

    def get_job_nodes(self, job, wait=2):
        """Get the hostname(s) of a job from the allocation
           Wait time is necessary because Slurm take about 3 seconds to
           register that a job has been submitted.

           :param Job job: A job instance
           :param int wait: time for wait before checking nodelist after job has been launched
                            defaults to 5 seconds.
           :returns: list of hostnames given a job or dict of job_name -> nodelist
           :raises: SSConfigError if called when using local launcher
           :raises: SmartSimError if job argument is not a Job object
        """
        self.check_local("Controller.get_job_nodes()")

        if not isinstance(job, Job):
            raise SmartSimError(
                "Argument must be a Job instance"
                )
        else:
            time.sleep(wait)
            nodes = self._get_job_nodes(job)
            return nodes

    def poll(self, interval, poll_db, verbose):
        """Poll the running simulations and recieve logging
           output with the status of the job.

           :param int interval: number of seconds to wait before polling again
           :param bool poll_db: poll dbnodes for status as well and see
                                it in the logging output
           :param bool verbose: set verbosity
           :raises: SSConfigError if called when using local launcher
        """
        self.check_local("Controller.poll()")

        all_finished = False
        while not all_finished:
            time.sleep(interval)
            ignore_db = not poll_db
            all_finished = self._poll(ignore_db=ignore_db, verbose=verbose)

    def _poll(self, ignore_db, verbose):
        """Poll all simulations and return a boolean for
           if all jobs are finished or not.

           :param bool verbose: set verbosity
           :param bool ignore_db: return true even if the orchestrator nodes are still running
           :returns: True or False for if all models have finished
        """
        finished = True
        for job in self._jobs().values():
            if ignore_db and job.entity.type == "db":
                continue
            else:
                self._check_job(job)
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
        self.check_local("Controller.finished()")
        try:
            if isinstance(entity, Orchestrator):
                raise SmartSimError("Finished() does not support Orchestrator instances")
            if not isinstance(entity, SmartSimEntity):
                raise SmartSimError("Finished() only takes arguments of SmartSimEntity instances")
            if isinstance(entity, Ensemble):
                return all([self.finished(model) for model in entity.models.values()])

            job = self._jobs[entity.name]
            self._check_job(job)
            return self._launcher.is_finished(job.status)
        except KeyError:
            raise SmartSimError(
                f"Entity by the name of {entity.name} has not been launched by this Controller")

    def get_orchestrator_status(self, orchestrator):
        """Return the workload manager status of an Orchestrator launched through the Controller

           :param orchestrator: The Orchestrator instance to check the status of
           :type orchestrator: Orchestrator instance
           :returns: statuses of the orchestrator in a list
           :rtype: list of str
        """
        if not isinstance(orchestrator, Orchestrator):
            raise SmartSimError(
                f"orchestrator argument was of type {type(orchestrator)} not of type Orchestrator")
        statuses = []
        for dbnode in orchestrator.dbnodes:
            statuses.append(self._get_status(dbnode))
        return statuses

    def get_ensemble_status(self, ensemble):
        """Return the workload manager status of an ensemble of models launched through the Controller

           :param ensemble: The Ensemble instance to check the status of
           :type ensemble: Ensemble instance
           :returns: statuses of the ensemble in a list
           :rtype: list of str
        """
        if not isinstance(ensemble, Ensemble):
            raise SmartSimError(
                f"ensemble argument was of type {type(ensemble)} not of type Ensemble")
        statuses = []
        for model in ensemble.models.values():
            statuses.append(self._get_status(model))
        return statuses

    def get_model_status(self, model):
        """Return the workload manager status of a model.

           :param model: the model to check the status of
           :type model: NumModel
           :returns: status of the model given by the workload manager
           :rtype: str
        """
        if not isinstance(model, NumModel):
            raise SmartSimError(
                f"model argument was of type {type(model)} not of type NumModel"
            )
        return self._get_status(model)

    def get_node_status(self, node):
        """Return the workload manager status of a SmartSimNode.

           :param node: the SmartSimNode to check the status of
           :type model: SmartSimNode
           :returns: status of the SmartSimNode given by the workload manager
           :rtype: str
        """
        if not isinstance(node, SmartSimNode):
            raise SmartSimError(
                f"node argument was of type {type(node)} not of type SmartSimNode"
            )
        return self._get_status(node)


    def init_launcher(self, launcher):
        """Run with a specific type of launcher"""
        if launcher is not None:
            # Init Slurm Launcher wrapper
            if launcher == "slurm":
                self._launcher = SlurmLauncher()
            # Run all ensembles locally
            elif launcher == "local":
                self._launcher = LocalLauncher()
            else:
                raise SSUnsupportedError("Launcher type not supported: " +
                                         launcher)
        else:
            raise SSConfigError(
                "Must provide a 'launcher' argument")

    def _stop_ensembles(self, ensembles):
        """Stops specified ensembles.  If ensembles is None,
           the function returns without performning any action.
           :param ensembles: List of ensembles to be stopped
           :type ensembles: list of ensemble, optional ensemble
           :raises: SmartSimError
        """
        if not ensembles:
            return

        if not all(isinstance(x, Ensemble) for x in ensembles):
            raise SmartSimError(
                "Only objects of type ensemble expected for input variable ensembles"
            )

        for ensemble in ensembles:
            models = list(ensemble.models.values())
            self._stop_models(models)

    def _stop_models(self, models):
        """Stops the specified models.  If the models is None,
           the function returns without performing any action.

           :param models: List of models to be stopped
           :type models: list of Model, option Model
           :raises: SmartSimError
        """

        if not models:
            return

        if not all(isinstance(x, NumModel) for x in models):
            raise SmartSimError(
                "Only objects of type NumModel expected for input variable models")

        for model in models:
            job = self._jobs[model.name]
            self._check_job(job)
            logger.info("Stopping model " + model.name + " job " +
                        job.get_job_id())
            self._launcher.stop(job.get_job_id())

    def _stop_nodes(self, nodes):
        """Stops specified nodes.  If nodes is None,
           the function returns without performning any action.

           :param nodes: List of nodes to be stopped
           :type nodes: list of SmartSimNode, optional SmartSimNode
           :raises: SmartSimError
        """
        if not nodes:
            return

        if not all(isinstance(x, SmartSimNode) for x in nodes):
            raise SmartSimError(
                "Only objects of type SmartSimNode expected for input variable nodes"
            )

        for node in nodes:
            job = self._jobs[node.name]
            self._check_job(job)
            logger.info("Stopping node " + node.name + " job " +
                        job.get_job_id())
            self._launcher.stop(job.get_job_id())


    def _stop_orchestrator(self, dbnodes):
        """Stops the orchestrator jobs that are currently running.

           :param dbnodes: the databases that make up the orchestrator
           :type dbnodes: a list of DBNode instances
           :raises: SmartSimError
        """
        if not dbnodes:
            return

        if not all(isinstance(x, DBNode) for x in dbnodes):
            raise SmartSimError(
                "Only objects of type DBNode expected for input variable dbnodes"
            )

        for dbnode in dbnodes:
            job = self._jobs[dbnode.name]
            self._check_job(job)
            logger.debug("Stopping orchestrator on job " + job.get_job_id())
            self._launcher.stop(job.get_job_id())


    def _prep_nodes(self, nodes):
        """Add the nodes to the list of requirement for all the entities
           to be launched.
        """
        if not nodes:
            return

        for node in nodes:
            run_settings, cmd = self._build_run_dict(node.run_settings)
            node.update_run_settings(run_settings)
            node.set_cmd(cmd)
            self._alloc_handler._add_to_allocs(run_settings)

    def _prep_orchestrator(self, orchestrator):
        """Add the orchestrator to the allocations requested"""
        if not orchestrator:
            return

        if isinstance(self._launcher, LocalLauncher):
            raise SSConfigError(
                "Orchestrators are not supported when launching locally"
            )

        for dbnode in orchestrator.dbnodes:
            dbnode_settings, cmd = self._build_run_dict(dbnode.get_run_settings())
            dbnode.update_run_settings(dbnode_settings)
            dbnode.set_cmd(cmd)
            self._alloc_handler._add_to_allocs(dbnode_settings)

    def _prep_ensembles(self, ensembles):
        """Add the models of each ensemble to the allocations requested"""
        if not ensembles:
            return

        for ensemble in ensembles:
            run_settings = {}
            cmd = ""
            if ensemble.name != "default":
                run_settings, cmd = self._build_run_dict(
                    ensemble.get_run_settings())
            for model in ensemble.models.values():
                if ensemble.name == "default":
                    run_settings, cmd = self._build_run_dict(
                        model.get_run_settings())
                model.update_run_settings(run_settings)
                model.set_cmd(cmd)
                self._alloc_handler._add_to_allocs(run_settings)

    def _remove_smartsim_args(self, arg_dict):
        ss_args = ["exe_args", "run_args", "executable", "run_command"]
        new_dict = dict()
        for k, v in arg_dict.items():
            if not k in ss_args:
                new_dict[k] = v
        return new_dict

    def _build_cmd(self, run_settings):
        """Contruct the run command from the entity run_settings"""

        exe = get_config("executable", run_settings, none_ok=False)
        exe_args = get_config("exe_args", run_settings, none_ok=True)
        if not exe_args:
            exe_args = ""
        cmd = " ".join((exe, exe_args))

        if isinstance(self._launcher, LocalLauncher):
            run_command = get_config("run_command", run_settings, none_ok=False)
            run_args = get_config("run_args", run_settings, none_ok=True)

            if not run_args:
                run_args = ""
            cmd = " ".join((run_command, run_args, exe, exe_args))

        return [cmd]

    def _build_run_dict(self, entity_info):
        """Build up the run settings by retrieving the settings of the Controller
           as well as the run_settings for each entity.

           :param dict entity_info: dictionary of settings for an entity
        """
        run_dict = {}
        try:
            # ensemble level values optional because there are defaults
            run_dict["nodes"] = get_config("nodes", entity_info, none_ok=True)
            run_dict["ppn"] = get_config("ppn", entity_info, none_ok=True)
            run_dict["partition"] = get_config("partition", entity_info, none_ok=True)
            cmd = self._build_cmd(entity_info)

            # set partition to default if none selected
            if not run_dict["partition"]:
                run_dict["partition"] = "default"

            return run_dict, cmd

        except KeyError as e:
            raise SSConfigError(
                "SmartSim could not find following required field: %s" %
                (e.args[0])) from e

    def _validate_allocations(self):
        """Validate the allocations with specific requirements provided by the user."""
        for partition, nodes in self._alloc_handler.partitions.items():
            if partition == "default":
                partition = None
            self._launcher.validate(nodes=nodes[0],
                                    ppn=nodes[1],
                                    partition=partition)

    def _get_allocations(self, duration):
        """Validate and retrieve n allocations where n is the number of partitions
           needed by the user.
        """
        if not isinstance(self._launcher, LocalLauncher):
            self._validate_allocations()

            for partition, nodes in self._alloc_handler.partitions.items():
                launch_partition = partition
                if partition == "default":
                    launch_partition = None
                alloc_id = self._launcher.get_alloc(
                    nodes=nodes[0],
                    ppn=nodes[1],
                    partition=launch_partition,
                    duration=duration)

                self._alloc_handler.allocs[partition] = alloc_id

    def _launch(self, ensembles, nodes, orchestrator):
        """Launch all entities within experiment with the configured launcher"""

        # launch orchestrator and all dbnodes
        if orchestrator:
            for dbnode in orchestrator.dbnodes:
                try:
                    self._launch_on_alloc(dbnode, orchestrator)
                    job_nodes = self.get_job_nodes(self._jobs[dbnode.name])
                    orchestrator.junction.store_db_addr(job_nodes[0],
                                                        orchestrator.port)
                except LauncherError as e:
                    logger.error(
                        "An error occured when launching the KeyDB nodes\n" +
                        "Check database node output files for details.")
                    raise SmartSimError(
                        "Database node %s failed to launch" % dbnode.name
                        ) from e

            # Create KeyDB cluster, min nodes for cluster = 3
            if len(orchestrator.dbnodes) > 2:
                db_nodes = self._jobs.get_db_nodes()
                port = orchestrator.port
                create_cluster(db_nodes, port)
                check_cluster_status(db_nodes, port)

        # launch the SmartSimNodes and update job nodes
        if nodes:
            for node in nodes:
                try:
                    self._launch_on_alloc(node, orchestrator)
                    self.get_job_nodes(self._jobs[node.name])
                except LauncherError as e:
                    logger.error(
                        "An error occured when launching SmartSimNodes\n" +
                        "Check node output files for details.")
                    raise SmartSimError(
                        "SmartSimNode %s failed to launch" % node.name
                        ) from e


        # Launch ensembles and their respective models and update job nodes
        if ensembles:
            for ensemble in ensembles:
                for model in ensemble.models.values():
                    try:
                        run_settings = model.get_run_settings()
                        self._launch_on_alloc(model, orchestrator)
                        self.get_job_nodes(self._jobs[model.name])
                    except LauncherError as e:
                        logger.error(
                            "An error occured when launching model ensembles.\n" +
                            "Check model output files for details.")
                        raise SmartSimError(
                            "Model %s failed to launch" % model.name) from e

    def _launch_on_alloc(self, entity, orchestrator):
        """launch a SmartSimEntity on an allocation provided by a workload manager."""

        cmd = entity.get_cmd()
        run_settings = self._remove_smartsim_args(entity.get_run_settings())

        if isinstance(self._launcher, LocalLauncher):
            pid = self._launcher.run(cmd, run_settings)
        else:
            # if orchestrator init and not a db node, setup connections to db
            if orchestrator and entity.type != "db":
                env_vars = orchestrator.get_connection_env_vars(entity.name)
                run_settings["env_vars"] = env_vars

            partition = run_settings["partition"]
            pid = self._launcher.run_on_alloc(
                cmd, self._alloc_handler.allocs[partition], **run_settings)

        logger.debug("Process id for " + entity.name + " is " + str(pid))
        self._jobs.add_job(entity.name, pid, entity)

    def _check_job(self, job):
        """Takes in job and sets job properties"""

        if not self._launcher:
            raise SmartSimError("No launcher set")
        job_id = job.get_job_id()
        status, returncode = self._launcher.get_job_stat(job_id)
        job.set_status(status, returncode)

    def _get_job_nodes(self, job):
        """Query the workload manager for the nodes that a particular job was launched on"""

        if job.nodes:
            return job.nodes
        else:
            job_id = job.get_job_id()
            nodes = self._launcher.get_job_nodes(job_id)
            job.nodes = nodes
            return nodes


    def _get_status(self, entity):
        """Return the workload manager given status of a job.

           :param entity: object launched by SmartSim. One of the following:
                          (SmartSimNode, NumModel, Orchestrator, Ensemble)
           :type entity: SmartSimEntity
           :returns: tuple of status
        """
        try:
            self.check_local("Controller._get_status()")
            if not isinstance(entity, SmartSimEntity):
                raise SmartSimError(
                    "Controller._get_status() only takes arguments of SmartSimEntity instances")
            job = self._jobs[entity.name]
            self._check_job(job)
        except KeyError:
            raise SmartSimError(
                f"Entity by the name of {entity.name} has not been launched by this Controller")
        return job.status

    def check_local(self, function_name):
        """Checks if the user called a function that isnt supported with the local launcher"""

        if isinstance(self._launcher, LocalLauncher):
            raise SSConfigError(f"{function_name} is not supported when launching locally")