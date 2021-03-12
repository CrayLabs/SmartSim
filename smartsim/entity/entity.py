import os.path as osp


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
        self.path = path

    @property
    def type(self):
        """Return the name of the class
        """
        return type(self).__name__

    def __repr__(self):
        return self.name
