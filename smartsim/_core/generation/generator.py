# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pathlib
import shutil
from distutils import dir_util
from os import mkdir, path, symlink

from ...entity import Model
from ...log import get_logger
from ..control import Manifest
from .modelwriter import ModelWriter

logger = get_logger(__name__)
logger.propagate = False


class Generator:
    """The primary job of the generator is to create the file structure
    for a SmartSim experiment. The Generator is responsible for reading
    and writing into configuration files as well.
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
        generator_manifest = Manifest(*args)
        self._gen_exp_dir()
        self._gen_orc_dir(generator_manifest.db)
        self._gen_entity_list_dir(generator_manifest.ensembles)
        self._gen_entity_dirs(generator_manifest.models)
        self._gen_entity_list_dir(generator_manifest.ray_clusters)

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

        if path.isfile(self.gen_path):
            raise FileExistsError(
                f"Experiment directory could not be created. {self.gen_path} exists"
            )
        if not path.isdir(self.gen_path):
            # keep exists ok for race conditions on NFS
            pathlib.Path(self.gen_path).mkdir(exist_ok=True)
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
            shutil.rmtree(orc_path, ignore_errors=True)
        pathlib.Path(orc_path).mkdir(exist_ok=True)

    def _gen_entity_list_dir(self, entity_lists):
        """Generate directories for EntityList instances

        :param entity_lists: list of EntityList instances
        :type entity_lists: list
        """

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
            elist.path = elist_dir

            self._gen_entity_dirs(elist.entities, entity_list=elist)

    def _gen_entity_dirs(self, entities, entity_list=None):
        """Generate directories for Entity instances

        :param entities: list of Entity instances
        :type entities: list
        :param entity_list: EntityList instance, defaults to None
        :type entity_list: EntityList, optional
        :raises EntityExistsError: if a directory already exists for an
                                   entity by that name
        """
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
                    raise FileExistsError(error)
            pathlib.Path(dst).mkdir(exist_ok=True)
            entity.path = dst
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
            to_write = []

            def _build_tagged_files(tagged):
                """Using a TaggedFileHierarchy, reproduce the tagged file
                directory structure

                :param tagged: a TaggedFileHierarchy to be built as a
                               directory structure
                :type tagged: TaggedFilesHierarchy
                """
                for file in tagged.files:
                    dst_path = path.join(entity.path, tagged.base, path.basename(file))
                    shutil.copyfile(file, dst_path)
                    to_write.append(dst_path)

                for dir in tagged.dirs:
                    mkdir(path.join(entity.path, tagged.base, path.basename(dir.base)))
                    _build_tagged_files(dir)

            _build_tagged_files(entity.files.tagged_hierarchy)

            # write in changes to configurations
            if isinstance(entity, Model):
                logger.debug(
                    f"Configuring model {entity.name} with params {entity.params}"
                )
                self._writer.configure_tagged_model_files(to_write, entity.params)

    def _copy_entity_files(self, entity):
        """Copy the entity files and directories attached to this entity.

        :param entity: SmartSimEntity
        :type entity: SmartSimEntity
        """
        if entity.files:
            for to_copy in entity.files.copy:
                dst_path = path.join(entity.path, path.basename(to_copy))
                if path.isdir(to_copy):
                    dir_util.copy_tree(to_copy, entity.path)
                else:
                    shutil.copyfile(to_copy, dst_path)

    def _link_entity_files(self, entity):
        """Symlink the entity files attached to this entity.

        :param entity: SmartSimEntity
        :type entity: SmartSimEntity
        """
        if entity.files:
            for to_link in entity.files.link:
                dst_path = path.join(entity.path, path.basename(to_link))
                symlink(to_link, dst_path)
