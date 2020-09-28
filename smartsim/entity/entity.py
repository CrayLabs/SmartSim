from os import getcwd
from os.path import join
from .files import EntityFiles
from ..error import SSConfigError
from ..utils.helpers import expand_exe_path

class SmartSimEntity:
    def __init__(self, name, path, entity_type, run_settings):
        """SmartSimEntity is the base class for all entities within SmartSim.
           Each entity starts of with default run settings for execution with
           a launcher. Upon user initialization, those defaults are
           updated with the arguments passed in by the user, and transformed
           into the commands for a specific type of launcher - slurm, local etc.
        """

    def __init__(self, name, path, entity_type, run_settings):
        """Initialize a SmartSim entity.

           Each entity must have a name, path, type, and
           run_settings. All entities within SmartSim
           share these attributes.

        :param name: Name of the entity
        :type name: str
        :param path: path to the filesystem location where output, conf,
                     and error files should be written.
        :type path: str
        :param entity_type: type of the entity
        :type entity_type: str
        :param run_settings: Arguments for the launcher specific to this
                             entity
        :type run_settings: dict
        """
        self.name = name
        self.type = entity_type
        self.run_settings = run_settings
        self.incoming_entities = []
        self._key_prefixing_enabled = False
        self._init_run_settings(path)
        self.files = None

    def _init_run_settings(self, init_path):
        """Initialize the run_settings from the defaults

        :param init_path: path to output, err and conf files,
                          defaults to os.getcwd()
        :type init_path: str
        """
        default_run_settings = {
            "nodes": 1,
            "ppn": 1,
        }

        new_path = getcwd()
        if init_path:
            new_path = init_path

        # expand the executable path
        self.run_settings["executable"] = self._expand_entity_exe()

        self.set_path(new_path)
        default_run_settings.update(self.run_settings)
        self.run_settings = default_run_settings

    def update_run_settings(self, updated_run_settings):
        """Update the run settings of an entity.

           This is commonly used with the Generator and
           Controller classes for changing the entity run_settings.

        :param updated_run_settings: new run settings
        :type updated_run_settings: dict
        """
        old_path = self.path
        self.run_settings.update(updated_run_settings)
        self.set_path(old_path)

    def set_path(self, new_path):
        """Set the path to error, output, and execution location

        :param new_path: path to set within run_settings
        :type new_path: str
        """
        self.path = new_path
        self.run_settings["cwd"] = self.path
        self.run_settings["out_file"] = join(self.path, self.name + ".out")
        self.run_settings["err_file"] = join(self.path, self.name + ".err")

    def register_incoming_entity(self, incoming_entity, receiving_client_type):
        """Register future communication between entities.

           Registers the named data sources that this entity
           has access to by storing the key_prefix associated
           with that entity

           Only python clients can have multiple incoming connections

           :param incoming_entity: The named SmartSim entity that data will be
                                   received from
           :param incoming_entity: SmartSimEntity
           :param receiving_client_type: The language of the SmartSim client used by
                                         this object. Can be cpp, fortran, python
           :param receiving_client_type: str
        """
        # Update list as clients are developed
        multiple_conn_supported = receiving_client_type in ['python']
        if not multiple_conn_supported and self.incoming_entities:
            raise SSConfigError(f"Receiving client of type '{receiving_client_type}'"+
                    " does not support multiple incoming connections")
        if incoming_entity.name in [in_entity.name for in_entity in
                                    self.incoming_entities]:
            raise SSConfigError(f"'{incoming_entity.name}' has already" +
                                "been registered as an incoming entity")

        self.incoming_entities.append(incoming_entity)

    def get_run_setting(self, setting):
        """Retrieve a setting from entity.run_settings

        :param setting: key for run_setting
        :type setting: str
        :return: run_setting value
        """
        return self.run_settings.get(setting)

    def enable_key_prefixing(self):
        """If called, the entity will prefix its keys with its own model name
        """
        self._key_prefixing_enabled = True

    def disable_key_prefixing(self):
        """If called, the entity will not prefix its keys with its own model name
        """
        self._key_prefixing_enabled = False

    def query_key_prefixing(self):
        """Inquire as to whether this entity will prefix its keys with its name"""
        return self._key_prefixing_enabled

    def attach_generator_files(self, to_copy=[], to_symlink=[], to_configure=[]):
        """Attach files to an entity for generation

           Attach files needed for the entity that, upon generation,
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
        self.files = EntityFiles(to_configure, to_copy, to_symlink)

    def _expand_entity_exe(self):
        """Run at initialization, this function finds and expands
           the executable for the entity.

        :param run_settings: dictionary of run_settings
        :type run_settings: dict
        """
        try:
            exe = self.run_settings["executable"]
            full_exe = expand_exe_path(exe)
            return full_exe
        except KeyError:
            raise SSConfigError(
                f"User did not provide an executable in the run settings for {self.name}")
        except SSConfigError as e:
            raise SSConfigError(
                f"Failed to create entity, bad executable argument in run settings") from e


    def __repr__(self):
        return self.name

    def __str__(self):
        entity_str = "Name: " + self.name + "\n"
        entity_str += "Type: " + self.type + "\n"
        entity_str += "run_settings = {\n"
        for param, value in self.run_settings.items():
            param = '"' + param + '"'
            if isinstance(value, str):
                value = '"' + value + '"'
            entity_str += " ".join((" ", str(param), ":", str(value), "\n"))
        entity_str += "}"
        return entity_str
