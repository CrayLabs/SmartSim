import sys

from .error import SSConfigError
from .helpers import read_config

class SSModule:

    def __init__(self, state, **kwargs):
        self.state = state
        self._init_args = kwargs
        self._get_targets()


    # change this to an internal method
    def log(self, message, level="info"):
        if level == "info":
            self.state.logger.info(message)
        elif level == "error":
            self.state.logger.error(message)
        else:
            self.state.logger.debug(message)

    def _get_targets(self):
        return self.state.targets

    def _get_exp_path(self):
        return self.state.get_experiment_path()


    def _get_config(self, conf_param):
        """Searches through init args and simulation.toml if the path
           is provided"""
        to_find = conf_param
        if isinstance(to_find, list):
            to_find = conf_param[-1]
            if to_find in self._init_args.keys():
                return self._init_args[to_find]
        # if not in init args search simulation.toml
        return self.state._get_toml_config(conf_param)

        