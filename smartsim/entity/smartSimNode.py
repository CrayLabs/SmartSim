import os
from os.path import join
from .entity import SmartSimEntity, EntityFiles
from ..error import SSUnsupportedError

class SmartSimNode(SmartSimEntity):
    def __init__(self, name, path, run_settings=dict()):
        super().__init__(name, path, "node", run_settings)

    def attach_generator_files(self, to_copy=[], to_symlink=[], to_configure=[]):
        """Attach files needed for the entity that, upon generation,
           will be located in the path of the entity.

           During generation files "to_copy" are just copied into
           the path of the entity, and files "to_symlink" are
           symlinked into the path of the entity.

           Files "to_configure" are text based model input files where
           parameters for the model are set. Note that only models
           support the "to_configure" field. These files must have
           fields tagged that correspond to the values the user
           would like to change. The tag is settable but defaults
           to a semicolon e.g. THERMO = ;10;

        :param to_copy: files to copy, defaults to []
        :type to_copy: list, optional
        :param to_symlink: files to symlink, defaults to []
        :type to_symlink: list, optional
        :param to_configure: [description], defaults to []
        :type to_configure: list, optional
        """
        if to_configure:
            error = "SmartSimNodes do not support reading and writing "\
                     "of configuration files."
            raise SSUnsupportedError(error)
        self.files = EntityFiles(to_configure, to_copy, to_symlink)
