import pickle
import sys
import zmq

from os import path, mkdir, listdir, getcwd
from .error import SmartSimError, SSConfigError
from .ensemble import Ensemble
from .model import NumModel
from .orchestrator import Orchestrator
from .smartSimNode import SmartSimNode
from .entity import SmartSimEntity
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
       :param str launcher: the method by which jobs are launched. options are "slurm", and "local"
    """
    def __init__(self, name, launcher="slurm"):
        self.name = name
        self.ensembles = []
        self.nodes = []
        self.orc = None
        self.exp_path = path.join(getcwd(), name)
        self._control = Controller()
        self._control.init_launcher(launcher)

    def start(self,
              ensembles=None,
              ssnodes=None,
              orchestrator=None,
              duration="1:00:00",
              run_on_existing=False):
        """Start the experiment with the created entities. If using a workload manager,
           'slurm' is the default, the total allocation size will be determined and
           aquired for the duration specified as an argument to this function.

           :param ensembles: Ensembles to launch with specified launcher
           :type ensembles: a list of Ensemble objects
           :param ssnodes: SmartSimNodes to launch with specified launcher
           :type ssnodes: a list of SmartSimNode objects
           :param orchestrator: Orchestrator object to be launched for entity communication
           :type orchestrator: Orchestrator object
           :param str duration: Time for the aquired allocation in format HH:MM:SS
           :param bool run_on_existing: If False, calculate and get an allocation
                                        for this workload
           :raises: SmartSimError
        """
        #TODO catch bad duration format
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
                orchestrator=orchestrator,
                duration=duration,
                run_on_existing=run_on_existing
            )
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

    def get_allocation(self, nodes, partition, duration="1:00:00", add_opts=None):
        """Get an allocation with the requested resources and return the allocation
           ID from the workload manager. Allocation obtained is automatically added
           to the smartsim allocations list for future calls to start to run on.

           :param int nodes: number of nodes in the requested allocation
           :param str partition: partition for the requested allocation
           :param str duration: duration of the allocation in format HH:MM:SS
           :param add_opts: additional options to pass to workload manager command
                            for obtaining an allocation
           :type add_opts: list of strings
           :return: allocation ID
           :rtype: string
           :raises: SmartSimError
        """
        try:
            alloc_id = self._control.get_allocation(
                nodes,
                partition,
                duration=duration,
                add_opts=add_opts
            )
            return alloc_id
        except SmartSimError as e:
            logger.error(e)
            raise e

    def add_allocation(self, alloc_id, partition, nodes):
        """Add an existing allocation to SmartSim so that future calls to start can run
           on it.

           :param str alloc_id: id of the allocation given by the workload manager
           :param str partition: partition of the allocation
           :param int nodes: number of nodes in the allocation
           :raises: SmartSimError
        """
        try:
            self._control.add_allocation(alloc_id, partition, nodes)
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
                orchestrator=self.orch
                )
        except SmartSimError as e:
            logger.error(e)
            raise

    def release(self, alloc_id=None):
        """Release the allocation(s) stopping all jobs that are currently running
           and freeing up resources. If an allocation ID is provided, only stop
           that allocation and remove it from SmartSim.

           :param str alloc_id: if provided, release that specific allocation
           :raises: SSConfigError if called when using local launcher
        """
        try:
            self._control.release(alloc_id=alloc_id)
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
            if isinstance(sender, SmartSimEntity):
                sender = sender.name
            if isinstance(reciever, SmartSimEntity):
                reciever = reciever.name
            if not isinstance(sender, str) or not isinstance(reciever, str):
                raise SmartSimError(
                    "Arguments to register connection must either be a str or a SmartSimEntity")
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
            if isinstance(name, SmartSimEntity):
                name = name.name
            if not isinstance(name, str):
                raise SmartSimError("Argument to delete_ensemble must be of type str")
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
            if not isinstance(ensemble, str):
                raise SmartSimError("Ensemble argument to get_model must be of type str")
            if not isinstance(model, str):
                raise SmartSimError("Model argument to get_model must be of type str")

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
            if not isinstance(ensemble, str):
                raise SmartSimError("Argument to get_ensemble must be of type str")
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
            if not isinstance(node, str):
                raise SmartSimError("Argument to get_node must be of type str")
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

           :returns: tcp address of orchestrator
           :rtype: returns a list if clustered orchestrator
           :raises: SSConfigError
           :raises: SmartSimError
        """
        if not self.orc:
            raise SSConfigError("No orchestrator has been initialized")
        addresses = []
        for dbnode in self.orc.dbnodes:
            job = self._control.get_job(dbnode.name)
            if not job.nodes:
                raise SmartSimError("Database has not been launched yet")
            for address in job.nodes:
                addr = ":".join((address, str(dbnode.port)))
                addresses.append(addr)
        if len(addresses) < 1:
            raise SmartSimError("Could not find nodes Database was launched on")
        return addresses

    def init_remote_launcher(self, launcher="slurm", addr="127.0.0.1", port=5555):
        """Initialize a remote launcher so that SmartSim can be run on compute
           nodes where slurm and other workload manager commands are not available.
           Remote launchers must be located on the same system, and be reachable
           via tcp.

           :param str launcher: Workload manager, default is "slurm" (currently only option)
           :param str addr: Address of the running remote command center
           :param int port: port the command center is running on
           :raises: SSConfigError
        """
        self._control.init_launcher(launcher=launcher, remote=True, addr=addr, port=port)

        # test to see if cmd_center has been spun up
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.setsockopt(zmq.LINGER, 0)
        address = "tcp://" + ":".join((addr, str(port)))
        socket.connect(address)
        try:
            socket.send(pickle.dumps((["PING"], None, None, None, None)))

            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            if poller.poll(1*1000):
                msg = socket.recv()
                logger.debug(pickle.loads(msg))
            else:
                raise SmartSimError(
                    f"Could not find command center at address {address}")
        except zmq.error.Again:
            raise SmartSimError(
                f"Could not find command center at address {address}")
        finally:
            socket.close()
            context.term()

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
