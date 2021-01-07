from os.path import join


class SmartSimEntity:
    def __init__(self, name, path, run_settings):
        """Initialize a SmartSim entity.

        Each entity must have a name, path, and
        run_settings. All entities within SmartSim
        share these attributes.

        :param name: Name of the entity
        :type name: str
        :param path: path to output, error, and configuration files
        :type path: str
        :param run_settings: Launcher settings specified in the experiment
                             entity
        :type run_settings: dict
        """
        self.name = name
        self.run_settings = run_settings
        self.set_path(path)

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

    @property
    def type(self):
        """Return the name of the class
        """
        return type(self).__name__

    def __repr__(self):
        return self.name
