
from error.errors import SSConfigError
from helpers import read_config

class SSModule:

    def __init__(self, state, config):
        self.state = state
        self._get_local_config(config)
        self._get_targets()

    def get_config(self, path):
        """Searches through the two levels of configuration for the
           SS library. ss-config.toml is searched first, followed by
           the config file specific to that module. Anything in
           ss-config.toml will override the module specific config.

           Args
             path (list): a list of strings containing path to config

           Returns
             a configuration value or error is one is not present.
        """
        # Search global configuration file
        top_level = self._search_config(path, self.state.config)
        if top_level:
            return top_level
        else:
            # Search local configuration file
            local_level = self._search_config(path, self.local_config)
            if local_level:
                return local_level
            else:
                raise SSConfigError(self.state.get_state(),
                                    "Could not find config value for key: " + ".".join(path))

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
            return None


    def _get_local_config(self, config):
        self.local_config = read_config(config)

    def _get_targets(self):
        targets = self.get_config(["execute", "targets"])
        self.targets = targets
