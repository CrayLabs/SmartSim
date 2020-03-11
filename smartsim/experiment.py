import pickle
import sys
from os import path, mkdir, listdir, getcwd
from .error import SmartSimError, SSConfigError
from .ensemble import Ensemble
from .model import NumModel
from .orchestrator import Orchestrator
from .smartSimNode import SmartSimNode
from .generation import Generator
from .control import Controller

from .utils import get_logger
logger = get_logger(__name__)


class Experiment:
    """In SmartSim, the Experiment class is an entity creation API that both houses
       and operates on the entities it creates. The entities that can be created are:

         - NumModels
         - Ensembles
         - SmartSimNodes
         - Orchestrator

        Each entity has a distinct purpose within an experiment.

        NumModel  - Instances of numerical models or "simulation" models. NumModels can
                    be created through a call to Experiment.create_model() and though
                    the creation of an Ensemble.
        Ensemble  - Ensembles are groups of NumModels to be either generated manually or
                    through a call to experiment.generate(). Ensembles can be given model
                    parameters to be written into input files for the model at runtime.
                    There are multiple ways of generating ensemble members; see
                    experiment.generate() for details.
        SmartSimNodes - Nodes run processes adjacent to the simulation. Nodes can be used
                        for anything from analysis, training, inference, etc. Nodes are the
                        most flexible entity with no requirements on language or framework.
                        Nodes are commonly used for acting on data being streamed out of a
                        simulation model through the orchestrator
        Orchestrator  - The Orchestrator is a KeyDB database, clustered or standalone, that
                        is launched prior to the simulation. The Orchestrator can be used
                        to store data from another entity in memory during the course of
                        an experiment. In order to stream data into the orchestrator or
                        recieve data from the orchestrator, one of the SmartSim clients
                        has to be used within a NumModel or SmartSimNode. Use
                        experiment.register_connection() to connect two entities within
                        SmartSim.

       :param str name: Name of the directory that will house all of
                        the created files and directories.
    """
    def __init__(self, name):
        self.name = name
        self.ensembles = []
        self.nodes = []
        self.orc = None
        self.exp_path = path.join(getcwd(), name)
        self._control = Controller()

    def start(self, launcher="slurm", duration="1:00:00"):
        """Start the experiment with the created entities. If using a workload manager,
           'slurm' is the default, the total allocation size will be determined and
           aquired for the duration specified as an argument to this function.

           :param str launcher: the type of launcher to use. default is "slurm".
                                options are "slurm" and "local"
           :param str duration: Time for the aquired allocation in format HH:MM:SS
           :raises: SmartSimError
        """
        #TODO catch bad duration format
        logger.info(f"Starting experiment: {self.name}")
        try:
            if not self._control._launcher:
                self._control.init_launcher(launcher)
            self._control.start(self.ensembles, self.nodes, self.orc)

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


    def stop_all(self):
        """Stop all entities that were created with this experiment

            :raises: SmartSimError
        """
        try:
            self._control.stop(
                ensembles=self.ensembles,
                nodes=self.nodes,
                orchestrator=self.orch
                )
        except SmartSimError as e:
            logger.error(e)
            raise

    def release(self):
        """Release the allocation(s) stopping all jobs that are currently running
           and freeing up resources.

           :raises: SmartSimError
        """
        try:
            self._control.release()
        except SmartSimError as e:
            logger.error(e)
            raise

    def poll(self, interval=10, poll_db=False, verbose=True):
        """Poll the running simulations and recieve logging output with the status
           of the job. If polling the database, jobs will continue until database
           is manually shutdown.

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


    def generate(self, model_files=None, node_files=None, tag=None, strategy="all_perm", **kwargs):
        """Generate the file structure for a SmartSim experiment. This includes the writing
           and configuring of input files for a model. Ensembles created with a 'params' argument
           will be expanded into multiple models based on a generation strategy. Model input files
           are specified with the model_files argument. All files and directories listed as strings
           in a list will be copied to each model within an ensemble. Every model file is read,
           checked for input variables to configure, and written. Input variables to configure
           are specified with a tag within the input file itself. The default tag is surronding
           an input value with semicolons. e.g. THERMO=;90;

           Files for SmartSimNodes can also be included to be copied into node directories but
           are never read nor written. All node_files will be copied into directories named after
           the name of the SmartSimNode within the experiment.

            :param model_files: The model files for the experiment.  Optional if model files
                                are not needed for execution.
            :type model_files: list of path like strings to directories or files
            :param node_files: files to be copied into node directories. These are most likely files
                               needed to run the node computations. e.g. a python script
            :type node_files: list of path like strings to directories or files

            :param str strategy: The permutation strategy for generating models within
                                ensembles.
                                Options are "all_perm", "random", "step", or a callable function.
                                Defaults to "all_perm"
            :raises: SmartSimError
        """
        try:
            generator = Generator()
            generator.set_strategy(strategy)
            if tag:
                generator.set_tag(tag)
            generator.generate_experiment(
                self.exp_path,
                ensembles=self.ensembles,
                nodes=self.nodes,
                orchestrator=self.orc,
                model_files=model_files,
                node_files=node_files,
                **kwargs
            )
        except SmartSimError as e:
            logger.error(e)
            raise

    def get_status(self, entity):
        """Get the status of a running job that was launched through a workload manager.
           Ensembles, Orchestrator, SmartSimNodes, and NumModel instances can all be passed
           to have their status returned as a string. The type of string and content will
           depend on the workload manager being used.

           :param entity: The SmartSimEntity object that was launched to check the status of
           :type entity: SmartSimEntity
           :returns: status of the entity
           :rtype: list if entity contains sub-entities such as cluster Orchestrator or Ensemble
           :raises: SmartSimError
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
                raise SmartSimError(
                    f"entity argument was of type {type(entity)} not SmartSimEntity")
        except SmartSimError as e:
            logger.error(e)
            raise


    def create_ensemble(self, name, params={}, run_settings={}):
        """Create a ensemble to be used within one or many of the SmartSim Modules. ensembles
           keep track of groups of models. Parameters can be given to a ensemble as well in
           order to generate models based on a combination of parameters and generation
           experimentgies. For more on generation strategies, see the Generator Class.

           :param str name: name of the new ensemble
           :param dict params: dictionary of model parameters to generate models from based
                               on a run strategy.
           :raises: SmartSimError

        """
        try:
            new_ensemble = None
            for ensemble in self.ensembles:
                if ensemble.name == name:
                    raise SmartSimError("A ensemble named " + ensemble.name +
                                        " already exists!")

            ensemble_path = path.join(self.exp_path, name)
            if path.isdir(ensemble_path):
                raise SmartSimError("ensemble directory already exists: " +
                                    ensemble_path)
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

    def create_model(self,
                     name,
                     ensemble="default",
                     params={},
                     path=None,
                     run_settings={}):
        """Create a model belonging to a specific ensemble. This function is
           useful for running a small number of models where the model files
           are already in place for execution.

           Calls to this function without specifying the `ensemble` argument
           result in the creation/usage a ensemble named "default", the default
           argument for `ensemble`.

           :param str name: name of the model to be created
           :param str ensemble: name of the ensemble to place model into
           :param dict params: dictionary of model parameters
           :param str path: (optional) path to model files, defaults to os.getcwd()
           :param dict run_settings: launcher settings for workload manager or local call
                                   e.g. {"ppn": 1, "nodes": 10, "partition":"default_queue"}
           :raises: SmartSimError
        """
        try:
            model_added = False
            model = NumModel(name, params, path, run_settings)
            if not path:
                path = getcwd()
            if ensemble == "default" and "default" not in [
                    ensemble.name for ensemble in self.ensembles]:

                # create empty ensemble
                self.create_ensemble(ensemble, params={}, run_settings={})
            for t in self.ensembles:
                if t.name == ensemble:
                    t.add_model(model)
                    model_added = True
            if not model_added:
                raise SmartSimError("Could not find ensemble by the name of: " +
                                    ensemble)
            return model
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_orchestrator(self,
                            path=None,
                            port=6379,
                            cluster_size=3,
                            partition=None):
        """Create an orchestrator database to faciliate the transfer of data
           for online training and inference. After the orchestrator is created,
           connections between models and nodes can be instantiated through a
           call to Experiment.register_connection().

           :param str path: desired path to output files of db cluster (defaults to
                            os.getcwd())
           :param int port: port for each database node for tcp communication
           :param int cluster_size: number of database nodes in cluster
           :param str partition: partition to launch db nodes
           :returns: Orchestrator object
           :raises: SmartSimError if one orchestrator has already been created
           """
        try:
            if self.orc:
                raise SmartSimError(
                    "Only one orchestrator can exist within a experiment.")
            orcpath = getcwd()
            if path:
                orcpath = path

            self.orc = Orchestrator(orcpath,
                                    port=port,
                                    cluster_size=cluster_size,
                                    partition=partition)
            return self.orc
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_node(self, name, script_path=None, run_settings={}):
        """Create a SmartSimNode for a specific task. Examples of SmartSimNode
           tasks include training, processing, and inference. Nodes can be used
           to run any task written in any language. The included script/executable
           for nodes often use the Client class to send and recieve data from
           the SmartSim orchestrator.

           :param str name: name of the node to be launched
           :param str script_path: path to the script or executable to be launched.
                                   (default is the current working directory of the
                                    SmartSim run script)
           :param dict run_settings: Settings for the workload manager can be set by
                                     including keyword arguments such as duration="1:00:00"
                                     or nodes=5
           :returns: SmartSimNode created
           :raises: SmartSimError if node exists by the same name
           """
        try:
            for node in self.nodes:
                if node.name == name:
                    raise SmartSimError("A node named " + node.name +
                                        " already exists!")
            node = SmartSimNode(name, script_path, run_settings=run_settings)
            self.nodes.append(node)
            return node
        except SmartSimError as e:
            logger.error(e)
            raise

    def register_connection(self, sender, reciever):
        """Create a runtime connection in orchestrator for data to be passed between two
           SmartSim entities. The possible types of connections right now are:

                Model -> Node
                Node  -> Node
                Node  -> Model

           :param str sender: name of the created entity with a Client instance to send
                              data to a registered counterpart
           :param str reciever: name of the created entity that will recieve data by
                                making calls to a Client instance.
           :raises: SmartSimError
        """
        try:
            if not self.orc:
                raise SmartSimError("Create orchestrator to register connections")
            else:
                # TODO check if sender and reciever are registered entities
                # TODO check for illegal connection types. e.g. model to model
                self.orc.junction.register(sender, reciever)
        except SmartSimError as e:
            logger.error(e)
            raise

    def delete_ensemble(self, name):
        """Delete a created ensemble from Experiment so that any future calls to SmartSim
           Modules will not include this ensemble.

           :param str name: name of the ensemble to be deleted
           :raises SmartSimError: if ensemble doesnt exist
        """
        try:
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

           :returns: NumModel instance
           :raises: SmartSimError if model is not found
        """
        try:
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

           :returns: ensemble instance
           :raises: SmartSimError
        """
        try:
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

           :returns: node instance
           :raises: SmartSimError
        """
        try:
            for n in self.nodes:
                if n.name == node:
                    return n
            raise SmartSimError("Node not found: " + node)
        except SmartSimError as e:
            logger.error(e)
            raise


    def __str__(self):
        experiment_str = "\n-- Experiment Summary --\n"
        if len(self.ensembles) > 0:
            experiment_str += "\n-- ensembles --"
            for ensemble in self.ensembles:
                experiment_str += str(ensemble)
        if len(self.nodes) > 0:
            experiment_str += "\n-- Nodes --"
            for node in self.nodes:
                experiment_str += str(node)
        if self.orc:
            experiment_str += str(self.orc)
        return experiment_str
