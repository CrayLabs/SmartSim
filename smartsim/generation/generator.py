import sys
import shutil

from itertools import product
from distutils import dir_util
from os import mkdir, getcwd, path, symlink

from ..database import Orchestrator
from ..entity import Model, Ensemble
from .modelwriter import ModelWriter
from ..error import EntityExistsError
from ..utils.entityutils import separate_entities
from ..error import SmartSimError, SSUnsupportedError, SSConfigError

from ..utils import get_logger

logger = get_logger(__name__)
logger.propagate = False


class Generator:
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

    def __init__(self, gen_path, overwrite=False):
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
        self.gen_path = gen_path
        self.overwrite = overwrite

    def generate_experiment(self, *args):
        """Run ensemble and experiment file structure generation

         TODO update this docstring
        Generate the file structure for a SmartSim experiment. This
        includes the writing and configuring of input files for a
        model.

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
        """
        entities, entity_lists, orchestrator = separate_entities(args)
        self._gen_exp_dir()
        self._gen_orc_dir(orchestrator)
        self._gen_entity_list_dir(entity_lists)
        self._gen_entity_dirs(entities)

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

    def _gen_exp_dir(self):
        """Create the directory for an experiment if it does not
        already exist.
        """

        if not path.isdir(self.gen_path):
            mkdir(self.gen_path)
        else:
            logger.info("Working in previously created experiment")

    def _gen_orc_dir(self, orchestrator):
        """Create the directory that will hold the error, output and
           configuration files for the orchestrator.

        :param orchestrator: Orchestrator instance
        :type orchestrator: Orchestrator
        """

        if not orchestrator:
            return

        orc_path = path.join(self.gen_path, "database")
        orchestrator.set_path(orc_path)

        # Always remove orchestrator files if present.
        if path.isdir(orc_path):
            shutil.rmtree(orc_path)
        mkdir(orc_path)

    def _gen_entity_list_dir(self, entity_lists):

        if not entity_lists:
            return

        for elist in entity_lists:

            elist_dir = path.join(self.gen_path, elist.name)
            if path.isdir(elist_dir):
                if self.overwrite:
                    shutil.rmtree(elist_dir)
                    mkdir(elist_dir)
            else:
                mkdir(elist_dir)

            self._gen_entity_dirs(elist.entities, entity_list=elist)

    def _gen_entity_dirs(self, entities, entity_list=None):
        if not entities:
            return

        for entity in entities:
            if entity_list:
                dst = path.join(self.gen_path, entity_list.name, entity.name)
            else:
                dst = path.join(self.gen_path, entity.name)

            if path.isdir(dst):
                if self.overwrite:
                    shutil.rmtree(dst)
                else:
                    error = (
                        f"Directory for entity {entity.name} "
                        f"already exists in path {dst}"
                    )
                    raise EntityExistsError(error)
            mkdir(dst)
            entity.set_path(dst)
            self._copy_entity_files(entity)
            self._link_entity_files(entity)
            self._write_tagged_entity_files(entity)

    def _write_tagged_entity_files(self, entity):
        """Read, configure and write the tagged input files for
           a Model instance within an ensemble. This function
           specifically deals with the tagged files attached to
           an Ensemble.

        :param entity: a SmartSimEntity, for now just Models
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
