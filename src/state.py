import logging

from helpers import read_config
from error.ss_error import SmartSimError, SSConfigError


class State:
    """Holds the state of the entire SS pipeline and the configurations
       necessary to run each stage of the pipeline"""

    def __init__(self, config=None):
        self.current_state = "Initializing"
        if config:
            self.config = config
        else:
            self.config = read_config()
        logging.info("SmartSim State: %s", self.current_state)


    def get_state(self):
        return self.current_state

    def update_state(self, new_state):
        self.current_state = new_state

    def get_config(self, key, config=None):
        """Retrieves a value from a toml file at an unspecified
           depth. Breadth first traversal of toml dict tree.

           Args
              Key (str): key being searched for
              Config (str): A sub-config file to search through

           Returns
              Value associated with key or a KeyError if key cannot
              be found.
        """
        value = self.search_config(key, self.config)
        if value:
            return value
        elif value == None and config != None:
            value = self.search_config(key, config)
            if value:
                return value
        raise SSConfigError(self.current_state,
                            "Missing key in configuration file: " + key)


    def search_config(self, key, config):
        visited = []
        try:
            for k, v in config.items():
                if k == key:
                    return v
                else:
                    if isinstance(v, dict):
                        visited.append(v)
            return self._search_config(key, visited)
        except KeyError:
            return None

    def _search_config(self, key, visited):
        if len(visited) == 0:
            raise KeyError
        else:
            cur_table = visited.pop()
            for k, v in cur_table.items():
                if k == key:
                    return v
                else:
                    if isinstance(v, dict):
                        visited.append(v)
            return self._search_config(key, visited)


