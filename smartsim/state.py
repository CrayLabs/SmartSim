import pickle
import sys
from os import path, mkdir, listdir, getcwd
from .error import SmartSimError, SSConfigError
from .ensemble import Ensemble
from .model import NumModel
from .orchestrator import Orchestrator
from .smartSimNode import SmartSimNode

from .utils import get_logger
logger = get_logger(__name__)


class State:
    """The State class holds most of the configurations necessary to run an
       experiment within SmartSim. State is responsible for the creation
       and storage of SmartSim entities such as models that are used to
       construct data pipelines.

       :param str experiment: Name of the directory that will house all of
                              the created files and directories.
    """
    def __init__(self, experiment):
        self.current_state = "Initializing"
        self.experiment = experiment
        self.ensembles = []
        self.nodes = []
        self.orc = None

    def __str__(self):
        state_str = "\n-- State Summary --\n"
        if len(self.ensembles) > 0:
            state_str += "\n-- ensembles --"
            for ensemble in self.ensembles:
                state_str += str(ensemble)
        if len(self.nodes) > 0:
            state_str += "\n-- Nodes --"
            for node in self.nodes:
                state_str += str(node)
        if self.orc:
            state_str += str(self.orc)
        return state_str

    def load_ensemble(self, name, ensemble_path=None):
        """Load a pickled ensemble into State for use. The ensemble currently must be from
           the same experiment it originated in. This can be useful if the experiment
           is being conducted over mutliple stages where execution does not all occur
           within the same script.

           :param str name: name of the pickled ensemble
           :param str ensemble_path: Path to the pickled ensemble. Defaults to os.getcwd()

        """
        try:
            tar_dir = path.join(getcwd(), self.experiment, name)
            if ensemble_path:
                tar_dir = ensemble_path
            if path.isdir(tar_dir):
                pickle_file = path.join(tar_dir, name + ".pickle")
                if path.isfile(pickle_file):
                    ensemble = pickle.load(open(pickle_file, "rb"))
                    if ensemble.experiment != self.experiment:
                        err = "ensemble must be loaded from same experiment \n"
                        msg = "ensemble experiment: {}   Current experiment: {}".format(
                            ensemble.experiment, self.experiment)
                        raise SmartSimError(err + msg)
                    self.ensembles.append(ensemble)
                else:
                    raise SmartSimError(
                        "ensemble, {}, could not be found".format(name))
            else:
                raise SmartSimError("ensemble directory could not be found!")
        except SmartSimError as e:
            logger.error(e)
            raise

    def create_ensemble(self, name, params={}, run_settings={}):
        """Create a ensemble to be used within one or many of the SmartSim Modules. ensembles
           keep track of groups of models. Parameters can be given to a ensemble as well in
           order to generate models based on a combination of parameters and generation
           stategies. For more on generation strategies, see the Generator Class.

           :param str name: name of the new ensemble
           :param dict params: dictionary of model parameters to generate models from based
                               on a run strategy.

        """
        new_ensemble = None
        try:
            for ensemble in self.ensembles:
                if ensemble.name == name:
                    raise SmartSimError("A ensemble named " + ensemble.name +
                                        " already exists!")

            ensemble_path = path.join(getcwd(), self.experiment, name)
            if path.isdir(ensemble_path):
                raise SmartSimError("ensemble directory already exists: " +
                                    ensemble_path)
            new_ensemble = Ensemble(name,
                                    params,
                                    self.experiment,
                                    ensemble_path,
                                    run_settings=run_settings)
            self.ensembles.append(new_ensemble)
        except SmartSimError as e:
            logger.error(e)
            raise
        return new_ensemble

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
        """
        model_added = False
        model = NumModel(name, params, path, run_settings)
        if not path:
            path = getcwd()
        if ensemble == "default" and "default" not in [
                ensemble.name for ensemble in self.ensembles
        ]:
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

    def create_orchestrator(self,
                            path=None,
                            port=6379,
                            cluster_size=3,
                            partition=None):
        """Create an orchestrator database to faciliate the transfer of data
           for online training and inference. After the orchestrator is created,
           connections between models and nodes can be instantiated through a
           call to State.register_connection().

           :param str path: desired path to output files of db cluster (defaults to
                            os.getcwd())
           :param int port: port for each database node for tcp communication
           :param int cluster_size: number of database nodes in cluster
           :param str partition: partition to launch db nodes o
           """
        if self.orc:
            raise SmartSimError(
                "Only one orchestrator can exist within a state.")
        if not path:
            orcpath = getcwd()

        self.orc = Orchestrator(orcpath,
                                port=port,
                                cluster_size=cluster_size,
                                partition=partition)

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
           """
        node = SmartSimNode(name, script_path, run_settings=run_settings)
        self.nodes.append(node)
        return node

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
        """
        if not self.orc:
            raise SmartSimError("Create orchestrator to register connections")
        else:
            # TODO check if sender and reciever are registered entities
            # TODO check for illegal connection types. e.g. model to model
            self.orc.junction.register(sender, reciever)

    def delete_ensemble(self, name):
        """Delete a created ensemble from State so that any future calls to SmartSim
           Modules will not include this ensemble.

           :param str name: name of the ensemble to be deleted
           :raises SmartSimError: if ensemble doesnt exist
        """
        # TODO delete the files as well if generated
        ensemble_deleted = False
        for t in self.ensembles:
            if t.name == name:
                self.ensembles.remove(t)
                ensemble_deleted = True
        if not ensemble_deleted:
            raise SmartSimError("Could not delete ensemble: " + name)

    def save(self):
        """Save each ensemble currently in state as a pickled python object.
           All models within the ensemble are maintained. ensembles can be reloaded
           into an experiment through a call to state.load_ensemble.
        """
        for ensemble in self.ensembles:
            pickle_path = path.join(ensemble.path, ensemble.name + ".pickle")
            if not path.isdir(ensemble.path):
                raise SmartSimError(
                    "ensembles must be generated in order to save them.  {0} does not exist."
                    .format(ensemble.path))
            file_obj = open(pickle_path, "wb")
            pickle.dump(ensemble, file_obj)
            file_obj.close()

    def get_model(self, model, ensemble):
        """Get a specific model from a ensemble.

           :param str model: name of the model to return
           :param str ensemble: name of the ensemble where the model is located

           :returns: NumModel instance
        """
        try:
            ensemble = self.get_ensemble(ensemble)
            model = ensemble[model]
            return model
        # if the ensemble is not found
        except SmartSimError:
            raise
        except KeyError:
            raise SmartSimError("Model not found: " + model)

    def get_ensemble(self, ensemble):
        """Return a specific ensemble from State

           :param str ensemble: Name of the ensemble to return

           :returns: ensemble instance
           :raises: SmartSimError
        """
        for t in self.ensembles:
            if t.name == ensemble:
                return t
        raise SmartSimError("ensemble not found: " + ensemble)

    def get_node(self, node):
        """Return a specific node from State

           :param str node: Name of the node to return

           :returns: node instance
           :raises: SmartSimError
        """
        for n in self.nodes:
            if n.name == node:
                return n
        raise SmartSimError("Node not found: " + node)

    def get_expr_path(self):
        expr_path = path.join(getcwd(), self.experiment)
        return expr_path
