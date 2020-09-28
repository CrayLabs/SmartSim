import sys
import shutil

from itertools import product
from os import mkdir, getcwd, path, symlink
from distutils import dir_util

from .modelwriter import ModelWriter
from ..entity import NumModel, Ensemble, SmartSimNode
from ..orchestrator import Orchestrator
from ..error import SmartSimError, SSUnsupportedError, SSConfigError
from ..error import GenerationError, EntityExistsError

from ..utils import get_logger
logger = get_logger(__name__)
logger.propagate = False


class Generator():
    """The primary job of the generator is to create, and configure models
       for ensembles. When a user creates an ensemble with parameters, the
       ensemble can be given to the generator for configuration of its model
       files. For more information on model generation see
       ``Generator.generate_experiment``.

       The Generator also creates the file structure for a SmartSim
       experiment. When called from experiment, all entities present
       within SmartSim will have directories created for their error
       and output files.
    """
    def __init__(self, overwrite=False):
        """Initialize a generator object

           if overwrite is true, replace any existing
           configured models within an ensemble if there
           is a name collision. Also replace any and all directories
           for the experiment with fresh copies. Otherwise, if overwrite
           is false, raises EntityExistsError when there is a name
           collision between entities.

        :param overwrite: toggle entity replacement, defaults to False
        :type overwrite: bool, optional
        """
        self._writer = ModelWriter()
        self.overwrite = overwrite

    def generate_experiment(self, exp_path, ensembles=[], nodes=[], orchestrator=None):
        """Run ensemble and experiment file structure generation

           Generate the file structure for a SmartSim experiment. This
           includes the writing and configuring of input files for a
           model. Ensembles created with a 'params' argument will be
           expanded into multiple models based on a generation strategy.

           To have files or directories present in the created entity
           directories, such as datasets or input files, call
           ``entity.attach_generator_files`` prior to generation. See
           ``entity.attach_generator_files`` for more information on
           what types of files can be included.

           Tagged model files are read, checked for input variables to
           configure, and written. Input variables to configure are
           specified with a tag within the input file itself.
           The default tag is surronding an input value with semicolons.
           e.g. ``THERMO=;90;``

        :param exp_path: path to the experiment directory
        :type exp_path: str
        :param ensembles: list of Ensemble instances
        :type ensembles: list
        :param nodes: list of SmartSimNode instances
        :type nodes: list
        :param orchestrator: orchestrator instance
        :type orchestrator: Orchestrator
        :raises SmartSimError: if generation fails
        """
        if isinstance(ensembles, Ensemble):
            ensembles = [ensembles]
        if isinstance(nodes, SmartSimNode):
            nodes = [nodes]
        if orchestrator and not isinstance(orchestrator, Orchestrator):
            raise TypeError(
                f"Argument given for orchestrator is of type {type(orchestrator)}, not Orchestrator"
            )
        self._create_experiment_dir(exp_path)
        self._create_orchestrator_dir(exp_path, orchestrator)
        self._create_nodes(exp_path, nodes)
        self._create_ensembles(exp_path, ensembles)

    def generate_ensemble(self, exp_path, ensembles):
        """Generate models and file structure for an ensemble

           Generate a single or a list of ensembles. This will
           generate the underlying model objects as well as configure
           and write their input files if attached.

           A directory for the ensemble and it's models will be
           created as a result of this function.

        :param exp_path: path to the experiment directory
        :type exp_path: str
        :param ensembles: list of Ensembles
        :type ensembles: list
        :raises TypeError: if ensembles argument is not of type Ensemble
                           or list of Ensembles
        """
        if isinstance(ensembles, Ensemble):
            ensembles = [ensembles]
        if not isinstance(ensembles[0], Ensemble):
            raise TypeError(
                f"ensembles argument must be of type Ensemble or list of Ensemble"
            )
        self._create_experiment_dir(exp_path)
        self._create_ensembles(exp_path, ensembles)

    def generate_node(self, exp_path, nodes):
        """Generate file structure for a SmartSimNode(s)

           Generate a single or list of SmartSimNodes.
           A directory will be created in the experiment
           directory for each model.

        :param exp_path: path to the experiment
        :type exp_path: str
        :param nodes: list of SmartSimNodes
        :type nodes: list
        :raises TypeError: if nodes is not of type SmartSimNode or
                           list of SmartSimNodes.
        """
        if isinstance(nodes, SmartSimNode):
            nodes = [nodes]
        if not isinstance(nodes[0], SmartSimNode):
            raise TypeError(
                f"nodes argument must be of type SmartSimNode or list of SmartSimNode"
            )
        self._create_experiment_dir(exp_path)
        self._create_nodes(exp_path, nodes)

    def set_tag(self, tag, regex=None):
        """Set the tag used for tagging input files

           Set a tag or a regular expression for the
           generator to look for when configuring new models.

           For example, a tag might be ``;`` where the
           expression being replaced in the model configuration
           file would look like ``;expression;``

           A full regular expression might tag specific
           model configurations such that the configuration
           files don't need to be tagged manually.

           :param tag: A string of characters that signify
                       an string to be changed. Defaults to ``;``
           :type tag: str
        """
        self._writer.set_tag(tag, regex)

    def _create_experiment_dir(self, exp_path):
        """Create the directory for an experiment if it does not
           already exist.

        :param exp_path: path to the experiment
        :type exp_path: str
        """

        if not path.isdir(exp_path):
            mkdir(exp_path)
        else:
            logger.info("Working in previously created experiment")


    def _create_orchestrator_dir(self, exp_path, orchestrator):
        """Create the directory that will hold the error, output and
           configuration files for the orchestrator.

        :param exp_path: path to the experiment
        :type exp_path: str
        :param orchestrator: Orchestrator instance
        :type orchestrator: Orchestrator
        """

        if not orchestrator:
            return

        orc_path = path.join(exp_path, "orchestrator")
        orchestrator.set_path(orc_path)

        # Always remove orchestrator files if present.
        if path.isdir(orc_path):
            shutil.rmtree(orc_path)
        mkdir(orc_path)

    def _create_nodes(self, exp_path, nodes):
        """Create the node directories and copy/symlink any listed
           files

        :param exp_path: path to the experiment
        :type exp_path: str
        :param nodes: nodes to generate directories for
        :type nodes: SmartSimNode
        :raises EntityExistsError: if node directory already exists
        """

        if not nodes:
            return

        for node in nodes:
            error = f"Node directory for {node.name} " \
                    f"already exists with {exp_path}"

            node_path = path.join(exp_path, node.name)
            node.set_path(node_path)
            if path.isdir(node_path):
                if not self.overwrite:
                    raise EntityExistsError(error)
                shutil.rmtree(node_path)
            mkdir(node_path)

            self._copy_entity_files(node)
            self._link_entity_files(node)

    def _create_ensembles(self, exp_path, ensembles):
        """Create the ensemble directories and the model directories
           within each ensemble.

        :param exp_path: path to the experiment
        :type exp_path: str
        :param ensembles: list of ensembles
        :type ensembles: list
        :raises EntityExistsError: if a model directory already exists
        """

        if not ensembles:
            return

        for ensemble in ensembles:

            ensemble_dir = path.join(exp_path, ensemble.name)
            if path.isdir(ensemble_dir):
                if self.overwrite:
                    shutil.rmtree(ensemble_dir)
                    mkdir(ensemble_dir)
            else:
                mkdir(ensemble_dir)

            for name, model in ensemble.models.items():
                dst = path.join(exp_path, ensemble.name, name)
                if path.isdir(dst):
                    if self.overwrite:
                        shutil.rmtree(dst)
                    else:
                        error = f"Model directory for {model.name} " \
                                f"already exists with {exp_path}"
                        raise EntityExistsError(error)
                mkdir(dst)
                model.set_path(dst)
                self._copy_entity_files(model)
                self._link_entity_files(model)
                self._write_tagged_entity_files(model)


    def _write_tagged_entity_files(self, entity):
        """Read, configure and write the tagged input files for
           a NumModel instance within an ensemble. This function
           specifically deals with the tagged files attached to
           an Ensemble.

        :param entity: a SmartSimEntity, for now just NumModels
        :type entity: SmartSimEntity
        """
        if entity.files:
            for i, tagged_file in enumerate(entity.files.tagged):
                dst_path = path.join(entity.path, path.basename(tagged_file))
                shutil.copyfile(tagged_file, dst_path)
                entity.files.tagged[i] = dst_path

            # write in changes to configurations
            if entity.type == "model":
                self._writer.configure_tagged_model_files(entity)

    def _copy_entity_files(self, entity):
         """Copy the entity files and directories attached to this entity.

         :param entity: SmartSimEntity
         :type entity: SmartSimEntity
         """
         if entity.files:
            for i, to_copy in enumerate(entity.files.copy):
                dst_path = path.join(entity.path, path.basename(to_copy))
                if path.isdir(to_copy):
                    dir_util.copy_tree(to_copy, entity.path)
                    entity.files.copy[i] = entity.path
                else:
                    shutil.copyfile(to_copy, dst_path)
                    entity.files.copy[i] = dst_path

    def _link_entity_files(self, entity):
        """Symlink the entity files attached to this entity.

         :param entity: SmartSimEntity
         :type entity: SmartSimEntity
         """
        if entity.files:
            for i, to_link in enumerate(entity.files.link):
                dst_path = path.join(entity.path, path.basename(to_link))
                symlink(to_link, dst_path)
                entity.files.link[i] = dst_path
