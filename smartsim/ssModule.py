import sys

from error.errors import SSConfigError
from helpers import read_config

class SSModule:

    def __init__(self, state):
        self.state = state
        self._get_targets()

    def get_config(self, path):
        """Searches for configurations in the simulation.toml

           Args
             path (list): a list of strings containing path to config

           Returns
             a configuration value or error is one is not present.
        """
        # Search global configuration file
        try:
            top_level = self._search_config(path, self.state.config)
            return top_level
        except SSConfigError as e:
            print(e)
            sys.exit()

    def _search_config(self, value_path, config):
        val_path = value_path.copy()
        # Helper method of get_config
        if val_path[0] in config.keys():
            if len(val_path) == 1:
                return config[val_path[0]]
            else:
                parent = val_path.pop(0)
                return self._search_config(val_path, config[parent])
        else:
            raise SSConfigError(self.state.get_state(),
                                "Could not find config value for key: " + ".".join(value_path))

    def _get_targets(self):
        # TODO adjust for "target" vs ["target1"] and ["target2"] in toml
        targets = self.get_config(["execute", "targets"])
        self.targets = targets
