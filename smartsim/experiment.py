import pickle
import sys
import zmq

from os import path, mkdir, listdir, getcwd, environ
from .error import SmartSimError, SSConfigError, EntityExistsError
from .orchestrator import Orchestrator
from .entity import SmartSimEntity, SmartSimNode, NumModel, Ensemble
from .generation import Generator
from .control import Controller
from .launcher import LocalLauncher

from .utils import get_logger
logger = get_logger(__name__)


class Experiment:
    """In SmartSim, the Experiment class is an entity creation API
       that both houses and operates on the entities it creates.
    """
    def __init__(self, name, launcher="slurm"):
        """Initialize an Experiment

           :param name: Name of the experiment
           :type name: str
           :param launcher: type of launcher, options are "local" and "slurm",
                            defaults to "slurm"
           :type launcher: str, optional
        """
        self.name = name
        self.ensembles = []
        self.nodes = []
        self.orc = None
        self.exp_path = path.join(getcwd(), name)
        self._control = Controller(launcher=launcher)

    def start(self, ensembles=None, ssnodes=None, orchestrator=None):
        """Start the SmartSim Experiment

           Start the experiment by turning all entities into jobs
           for the underlying launcher specified at experiment
           initialization. All entities in the experiment will be
           launched if not arguments are passed.

           :param ensembles: list of Ensembles, defaults to None
           :type ensembles: list Ensemble instances, optional
           :param ssnodes: list of SmartSimNodes, defaults to None
           :type ssnodes: list of SmartSimNode instances, optional
           :param orchestrator: Orchestrator instance, defaults to None
           :type orchestrator: Orchestrator
        """
        logger.info(f"Starting experiment: {self.name}")
        try:
            if not ensembles:
                ensembles = self.ensembles
            if not ssnodes:
                ssnodes = self.nodes
            if not orchestrator:
                orchestrator = self.orc

            self._control.start(
                ensembles=ensembles,
                nodes=ssnodes,
                orchestrator=orchestrator)
        except SmartSimError as e:
            logger.error(e)
            raise

    def stop(self, ensembles=None, models=None, nodes=None, orchestrator=None):
        """Stop specific entities launched through SmartSim. This method is only
           applicable when launching with a workload manager.

           :param ensembles: Ensemble objects to be stopped
           :type Ensembles: list of Ensemble objects
           :param models: Specific models to be stopped
           :type models: list of Model objects
           :param nodes: SmartSimNodes to be stopped
           :type nodes: list of SmartSimNodes
           :param orchestrator: the orchestrator to be stoppped
           :type orchestrator: instance of Orchestrator
           :raises: SmartSimError
        """
        try:
            self._control.stop(
                ensembles=ensembles,
                models=models,
                nodes=nodes,
                orchestrator=orchestrator
            )
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_allocation(self, nodes=1, ppn=1, duration="1:00:00", **kwargs):
        """Get an allocation through the launcher for future calls
           to start to launch entities onto. This allocation is
           maintained within SmartSim. To release the allocation
           call Experiment.release().

           The kwargs can be used to pass extra settings to the
           workload manager such as the following for Slurm:
             - nodelist="nid00004"

           For arguments without a value, pass None or and empty
           string as the value for the kwarg. For Slurm:
             - exclusive=None

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
        try:
            alloc_id = self._control.get_allocation(
                nodes=nodes,
                ppn=ppn,
                duration=duration,
                **kwargs
            )
            return alloc_id
        except SmartSimError as e:
            logger.error(e)
            raise e

    def add_allocation(self, alloc_id):
        """Add an allocation to SmartSim such that entities can
           be launched on it.

        :param alloc_id: id of the allocation from the workload manager
        :type alloc_id: str
        :raises SmartSimError: If the allocation cannot be found
        """
        try:
            self._control.add_allocation(alloc_id)
        except SmartSimError as e:
            logger.error(e)
            raise e

    def stop_all(self):
        """Stop all entities that were created with this experiment

            :raises: SmartSimError
        """
        try:
            self._control.stop(
                ensembles=self.ensembles,
                nodes=self.nodes,
                orchestrator=self.orc
                )
        except SmartSimError as e:
            logger.error(e)
            raise

    def release(self, alloc_id=None):
        """Release the allocation(s) stopping all jobs that are
           currently running and freeing up resources. If an
           allocation ID is provided, only stop that allocation
           and remove it from SmartSim.

        :param alloc_id: id of the allocation, defaults to None
        :type alloc_id: str, optional
        :raises SmartSimError: if fail to release allocation
        """
        try:
            self._control.release(alloc_id=alloc_id)
        except SmartSimError as e:
            logger.error(e)
            raise

    def poll(self, interval=10, poll_db=False, verbose=True):
        """Poll the running simulations and receive logging output
           with the status of the job. If polling the database,
           jobs will continue until database is manually shutdown.

           :param int interval: number of seconds to wait before polling again
           :param bool poll_db: poll dbnodes for status as well and see
                                it in the logging output
           :param bool verbose: set verbosity
           :raises: SmartSimError
        """
        try:
            self._control.poll(interval, poll_db, verbose)
        except SmartSimError as e:
            logger.error(e)
            raise


    def finished(self, entity):
        """Return a boolean indicating wether or not a job has finished.

           :param entity: object launched by SmartSim. One of the following:
                          (SmartSimNode, NumModel, Orchestrator, Ensemble)
           :type entity: SmartSimEntity
           :returns: bool
        """
        try:
            return self._control.finished(entity)
        except SmartSimError as e:
            logger.error(e)
            raise

    def generate(self, tag=None, strategy="all_perm", overwrite=False, **kwargs):
        """Generate the file structure for a SmartSim experiment. This
           includes the writing and configuring of input files for a
           model. Ensembles created with a 'params' argument will be
           expanded into multiple models based on a generation strategy.

           To have files or directories present in the created entity
           directories, such as datasets or input files, call
           entity.attach_generator_files prior to generation. See
           entity.attach_generator_files() for more information on
           what types of files can be included.

           Tagged model files are read, checked for input variables to
           configure, and written. Input variables to configure are
           specified with a tag within the input file itself.
           The default tag is surronding an input value with semicolons.
           e.g. THERMO=;90;

            :param str strategy: The permutation strategy for generating models within
                                ensembles.
                                Options are "all_perm", "random", "step", or a
                                callable function. Defaults to "all_perm"
            :raises SmartSimError: if generation fails
        """
        try:
            generator = Generator(overwrite=overwrite)
            generator.set_strategy(strategy)
            if tag:
                generator.set_tag(tag)
            generator.generate_experiment(
                self.exp_path,
                ensembles=self.ensembles,
                nodes=self.nodes,
                orchestrator=self.orc,
                **kwargs
            )
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_status(self, entity):
        """Get the status of a running job that was launched through
           a workload manager. Ensembles, Orchestrator, SmartSimNodes,
           and NumModel instances can all be passed to have their
           status returned as a string. The type of string and content
           will depend on the workload manager being used.

           :param entity: The SmartSimEntity object that was launched
                          to check the status of
           :type entity: SmartSimEntity
           :returns: status of the entity
           :rtype: list if entity contains sub-entities such as cluster
                   Orchestrator or Ensemble
           :raises SmartSimError: if status retrieval fails
           :raises TypeError: if one argument was not a SmartSimEntitiy
        """
        try:
            if isinstance(entity, Ensemble):
                return self._control.get_ensemble_status(entity)
            elif isinstance(entity, Orchestrator):
                return self._control.get_orchestrator_status(entity)
            elif isinstance(entity, NumModel):
                return self._control.get_model_status(entity)
            elif isinstance(entity, SmartSimNode):
                return self._control.get_node_status(entity)
            else:
                raise TypeError(
                    f"entity argument was of type {type(entity)} not SmartSimEntity")
        except SmartSimError as e:
            logger.error(e)
            raise


    def create_ensemble(self, name, params={}, run_settings={}, overwrite=False):
        """Create a ensemble to be used within one or many of the
           SmartSim Modules. Ensembles contain groups of models.
           Parameters can be given to a ensemble in order to generate
           models based on a combination of parameters and generation
           strategies. For more on generation strategies, see the
           Generator Class.

        :param name: name of the ensemble
        :type name: str
        :param params: model parameters for generation strategies,
                       defaults to {}
        :type params: dict, optional
        :param run_settings: define how the model should be run,
                             defaults to {}
        :type run_settings: dict, optional
        :param overwrite: overwrite an existing ensemble if on by the
                          same name already exists
        :type overwrite: bool
        :raises SmartSimError: If ensemble cannot be created
        :return: the created Ensemble
        :rtype: Ensemble
        """
        try:
            new_ensemble = None
            for ensemble in self.ensembles:
                if ensemble.name == name:
                    if overwrite:
                        self.ensembles.remove(ensemble)
                    else:
                        error = f"Ensemble {ensemble.name} already exists.\n"
                        error += f"Call with overwrite=True to replace this Ensemble"
                        raise EntityExistsError(error)

            ensemble_path = path.join(self.exp_path, name)
            new_ensemble = Ensemble(name,
                                    params,
                                    self.name,
                                    ensemble_path,
                                    run_settings=run_settings)
            self.ensembles.append(new_ensemble)

            return new_ensemble
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_model(self, name, ensemble="default", params={}, path=None,
                     run_settings={}, enable_key_prefixing=False, overwrite=False):
        """Create a model belonging to a specific ensemble. This function is
           useful for running a small number of models where the model files
           are already in place for execution.

           Calls to this function without specifying the `ensemble` argument
           result in the creation/usage a ensemble named "default", the default
           argument for `ensemble`.

           Models in the default ensemble will be launched with their specific
           run_settings as defined in intialization here. Otherwise the model
           will use the run_settings defined for the Ensemble

        :param name: name of the model
        :type name: str
        :param ensemble: name of the ensemble to add the model to,
                         defaults to "default"
        :type ensemble: str, optional
        :param params: model parameters for generation strategies,
                       defaults to {}
        :type params: dict, optional
        :param path: path to where the model should be executed at runtime,
                     defaults to os.getcwd()
        :type path: str, optional
        :param run_settings: defines how the model should be run,
                             defaults to {}
        :type run_settings: dict, optional
        :param enable_key_prefixing: If true, keys sent by this model will be
                                     prefixed with the model's name.
                                     Optional, defaults to False
        :type enable_key_prefixing: bool
        :param overwrite: replace model if one exists by the same name,
                          Optional, defaults to false
        :type overwrite: bool
        :raises SmartSimError: if ensemble name provided doesnt exist
        :return: the created model
        :rtype: NumModel
        """
        try:
            model_added = False
            model = NumModel(name, params, path, run_settings)
            if enable_key_prefixing:
                model._key_prefixing_enabled = True
            if not path:
                path = getcwd()
            if ensemble == "default" and "default" not in [
                    ensemble.name for ensemble in self.ensembles]:

                # create empty ensemble
                self.create_ensemble(ensemble, params={}, run_settings={})
            for t in self.ensembles:
                if t.name == ensemble:
                    t.add_model(model, overwrite=overwrite)
                    model_added = True
            if not model_added:
                raise SmartSimError(
                    f"Could not find ensemble {ensemble}")
            return model
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_orchestrator(self, path=None, port=6379, overwrite=False,
                            db_nodes=1, **kwargs):
        """Create an in-memory database to run with an experiment.

        Launched entities can communicate with the orchestrator through use
        of one of the Python, C, C++ or Fortran clients.

        With the default settings, this function can be used to create
        a local orchestrator that will run in parallel with other
        entities running serially in an experiment. If launching the
        orchestrator on a machine with a workload manager, include
        "alloc" as a kwarg to launch the orchestrator on a specified
        compute resource allocation.  For creating
        clustered orchestrators accross multiple compute nodes,
        set db_nodes to 3 or larger.  Additionally, the kwarg "dpn"
        can be used to launch multiple databases per compute node.

        :param path: desired path for orchestrator output/error, defaults to cwd
        :type path: str, optional
        :param port: port orchestrator should run on, defaults to 6379
        :type port: int, optional
        :param overwrite: flag to indicate that existing orcestrator files
        in the experiment directory should be overwritten
        :type overwrite: bool, optional
        :param db_nodes: number of database nodes in the cluster, defaults to 3
        :type db_nodes: int, optional
        :raises SmartSimError: if an orchestrator already exists
        :return: Orchestrator instance created
        :rtype: Orchestrator
        """
        try:
            if isinstance(self._control._launcher, LocalLauncher) and db_nodes>1:
                error = "Clustered orchestrators are not supported when using the local launcher\n"
                error += "Use Experiment.create_orchestrator() for launching an orchestrator"
                error += "with the local launcher"
                raise SmartSimError(error)

            if self.orc and not overwrite:
                error = "Only one orchestrator can exist within a experiment.\n"
                error += "Call with overwrite=True to replace the current orchestrator"
                raise EntityExistsError(error)

            orcpath = getcwd()
            if path:
                orcpath = path

            self.orc = Orchestrator(orcpath,
                                    port=port,
                                    db_nodes=db_nodes,
                                    **kwargs)
            return self.orc
        except SmartSimError as e:
            logger.error(e)
            raise

    def reconnect_orchestrator(self, previous_orc_dir):
        """Reconnect to an orchestrator that was created in a separate
        SmartSim experiment.

        :param previous_orc_dir: Directory where the previous experiment database
        files are located.
        :type orc_dir: str
        :raises SmartSimError: The database config file is missing, incomplete,
                               or corrupted
        :return: Orchestrator instance
        :rtype: Orchestrator
        """

        try:

            if isinstance(self._control._launcher, LocalLauncher):
                raise SmartSimError("Local launcher does not support "\
                                    "reconnecting to a database.")

            if self.orc:
                raise SmartSimError(
                    "Only one orchestrator can exist within a experiment.")

            db_file = "/".join((previous_orc_dir, "smartsim_db.dat"))
            if not path.exists(db_file):
                raise SmartSimError(f"The SmartSim database config file "\
                                    "{db_file} cannot be found.")

            try:
                with open(db_file, "rb") as pickle_file:
                    db_config = pickle.load(pickle_file)
            except (OSError, IOError) as e:
                raise SmartSimError(str(e))

            err_message = "The SmartSim database config file is incomplete.  "
            if not "orc" in db_config:
                raise SmartSimError(err_message +
                                    "Could not find the orchestrator object.")

            if not db_config["orc"].port:
                raise SmartSimError(err_message +
                                    "The orchestrator is missing db port "\
                                    "information.")

            if not db_config["orc"].dbnodes:
                raise SmartSimError(err_message +
                                    "The orchestrator is missing db node "\
                                    "information.")

            if not "db_jobs" in db_config:
                raise SmartSimError(err_message +
                                    "Could not find database job objects.")

            for db_job in db_config["db_jobs"].values():
                self._control._jobs.db_jobs[db_job.name] = db_job

            self.orc = db_config["orc"]

            if not isinstance(self._control._launcher, LocalLauncher):
                db_statuses = self.get_status(self.orc)
                if not all(status=="RUNNING" for status in db_statuses):
                    raise SmartSimError("The specified database is no "\
                                        "longer running")

            return self.orc

        except SmartSimError as e:
            logger.error(e)
            raise

    def create_node(self, name, path=None, run_settings={}, overwrite=False,
                    enable_key_prefixing=False):
        """Create a SmartSimNode for a specific task. Examples of SmartSimNode
           tasks include training, processing, and inference. Nodes can be used
           to run any task written in any language. The included script/executable
           for nodes often use the Client class to send and receive data from
           the SmartSim orchestrator.

           :param name: name of the node to be launched
           :type name: str
           :param path: path to the script or executable to be launched.
                        (default is the current working directory of the
                        SmartSim run script)
           :type path: str
           :param run_settings: Settings for the workload manager can be set by
                                including keyword arguments such as
                                duration="1:00:00" or nodes=5
           :type run_settings: dict
           :param enable_key_prefixing: If true, keys sent by this model will be
                                        prefixed with the model's name.
                                        Defaults to False
           :type enable_key_prefixing: bool, optional
           :param overwrite: replace node if one exists by the same name
           :raises: SmartSimError if node exists by the same name
           :returns: SmartSimNode created
           :rtype: SmartSimNode
        """
        try:
            for node in self.nodes:
                if node.name == name:
                    if overwrite:
                        self.nodes.remove(node)
                    else:
                        error = f"Node {node.name} already exists.\n"
                        error += "Call with overwrite=True to replace"
                        raise EntityExistsError(error)

            node = SmartSimNode(name, path, run_settings=run_settings)
            if enable_key_prefixing:
                node._key_prefixing_enabled = True
            self.nodes.append(node)
            return node
        except SmartSimError as e:
            logger.error(e)
            raise

    def delete_ensemble(self, name):
        """Delete a created ensemble from Experiment so that
           any future calls to SmartSim Modules will not include
           this ensemble.

           :param str name: name of the ensemble to be deleted
           :raises TypeError: if argument is not a str name of an ensemble
           :raises SmartSimError: if ensemble doesnt exist
        """
        try:
            if isinstance(name, SmartSimEntity):
                name = name.name
            if not isinstance(name, str):
                raise TypeError("Argument to delete_ensemble must be of type str")
            ensemble_deleted = False
            for t in self.ensembles:
                if t.name == name:
                    self.ensembles.remove(t)
                    ensemble_deleted = True
            if not ensemble_deleted:
                raise SmartSimError("Could not delete ensemble: " + name)
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_model(self, model, ensemble):
        """Get a specific model from a ensemble.

           :param str model: name of the model to return
           :param str ensemble: name of the ensemble where the model is located

           :raises SmartSimError: if model is not found
           :raises TypeError: if arguments are not str names of a model
                              and/or an Ensemble
           :returns: NumModel instance
           :rtype: NumModel
        """
        try:
            if not isinstance(ensemble, str):
                raise TypeError("Ensemble argument to get_model must be of type str")
            if not isinstance(model, str):
                raise TypeError("Model argument to get_model must be of type str")

            ensemble = self.get_ensemble(ensemble)
            model = ensemble[model]
            return model
        except KeyError:
            raise SmartSimError("Model not found: " + model)
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_ensemble(self, ensemble):
        """Return a specific ensemble from Experiment

           :param str ensemble: Name of the ensemble to return
           :raises SmartSimError: if ensemble is not found
           :raises TypeError: if argument is not a str name of
                              an ensemble
           :returns: ensemble instance
           :rtype: Ensemble
        """
        try:
            if not isinstance(ensemble, str):
                raise TypeError("Argument to get_ensemble must be of type str")
            for t in self.ensembles:
                if t.name == ensemble:
                    return t
            raise SmartSimError("ensemble not found: " + ensemble)
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_node(self, node):
        """Return a specific node from Experiment

           :param str node: Name of the node to return
           :raises SmartSimError: if node cannot be found
           :raises TypeError: if argument is not a str name
                    of an node
           :returns: node instance
           :rtype: SmartSimNode
        """
        try:
            if not isinstance(node, str):
                raise TypeError("Argument to get_node must be of type str")
            for n in self.nodes:
                if n.name == node:
                    return n
            raise SmartSimError("Node not found: " + node)
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_db_address(self):
        """Get the TCP address of the orchestrator returned by pinging the
           domain name used by the workload manager e.g. nid00004 returns
           127.0.0.1

           :raises SmartSimError: if orchestrator has not been launched
           :raises: SmartSimError: if database nodes cannot be found
           :returns: tcp address of orchestrator
           :rtype: returns a list if clustered orchestrator
        """
        if not self.orc:
            raise SmartSimError("No orchestrator has been initialized")
        addresses = []
        for dbnode in self.orc.dbnodes:
            job = self._control._jobs[dbnode.name]
            if not job.nodes:
                raise SmartSimError("Database has not been launched yet.")

            for address in job.nodes:
                for port in dbnode.ports:
                    addr = ":".join((address, str(port)))
                    addresses.append(addr)
        if len(addresses) < 1:
            raise SmartSimError("Could not find nodes Database was launched on")
        return addresses

    def __str__(self):
        experiment_str = "\n-- Experiment Summary --\n"
        if len(self.ensembles) > 0:
            experiment_str += "\n-- ensembles --\n"
            for ensemble in self.ensembles:
                experiment_str += str(ensemble)
        if len(self.nodes) > 0:
            experiment_str += "\n-- Nodes --\n"
            for node in self.nodes:
                experiment_str += str(node)
        if self.orc:
            experiment_str += "\n -- Orchestrator --\n"
            experiment_str += str(self.orc)
        return experiment_str
