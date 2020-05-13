from os import getcwd
from os.path import join
from smartsim.error.errors import SSConfigError

class SmartSimEntity:
    """SmartSimEntity is the base class for all entities within SmartSim.
       Each entity starts of with default run settings for launching with
       a workload manager. Upon user initialization, those defaults are
       updated with the arguments passed in by the user, and transformed
       into the commands for a specific type of launcher - slurm, local etc.
    """

    def __init__(self, name, path, entity_type, run_settings):
        """Intialize a SmartSim entity. Each entity must have a name,
           path, type, and run_settings. All entities within SmartSim
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
        self._init_run_settings(path)

    def _init_run_settings(self, init_path):
        """intialize the run_settings from the defaults

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

        self.set_path(new_path)
        default_run_settings.update(self.run_settings)
        self.run_settings = default_run_settings

    def update_run_settings(self, updated_run_settings):
        """Update the run settings of an entity. This is commonly
           used with the Generator and Controller classes for changing
           the entity run_settings.

        :param updated_run_settings: new run settings
        :type updated_run_settings: dict
        """
        old_path = self.path
        self.run_settings.update(updated_run_settings)
        self.set_path(old_path)

    def set_path(self, new_path):
        """Set the path to error, output, and command location within
           the filesystem.

        :param new_path: path to set within run_settings
        :type new_path: str
        """
        self.path = new_path
        self.run_settings["cwd"] = self.path
        self.run_settings["out_file"] = join(self.path, self.name + ".out")
        self.run_settings["err_file"] = join(self.path, self.name + ".err")

    def get_run_setting(self, setting):
        """retrieve a run_setting

        :param setting: key for run_setting
        :type setting: str
        :return: run_setting value
        """
        return self.run_settings.get(setting)

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