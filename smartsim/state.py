import logging
import sys
from os import path, mkdir
from .helpers import read_config, get_SSHOME
from .error import SmartSimError, SSConfigError
from .target import Target


class State:
    """Holds the state of the entire SS pipeline and the configurations
       necessary to run each stage of the pipeline"""

    def __init__(self, experiment=None, config=None, log_level="DEV"):
        self.current_state = "Initializing"
        self._config = read_config(config)
        self.targets = []
        self.__set_experiment(experiment)
        self.__create_logger(log_level)
        self.__load_targets()


    def get_state(self):
        return self.current_state

    def get_experiment_path(self):
        return path.join(get_SSHOME(), self.experiment)

    def create_target(self, target):
        try:
            if not isinstance(target, Target):
                raise SmartSimError(self.current_state, "Input to create_target() must be a Target Instance")
            if not self.targets:
                self.targets = [target]
            else:       
                self.targets.append(target)
        except SmartSimError as e:
            print(e)
            sys.exit()

    def _set_state(self, new_state):
        self.current_state = new_state

    def __set_experiment(self, experiment_name):
        if not experiment_name:
            try:
                self.experiment = self._get_toml_config(["model", "experiment"])
            except SSConfigError:
                print("Experiment name must be defined in either simulation.toml or in state initialization")
                sys.exit()
        else:
            self.experiment = experiment_name

        
    def __load_targets(self):
        """Load targets if they are present within the simulation.toml"""
        if self._config:
            try:
                model_targets = self._get_toml_config(["model", "targets"])
                for target in model_targets:
                    param_dict = self._get_toml_config([target])
                    target_path = path.join(get_SSHOME(), self.experiment, target)
                    new_target = Target(target, param_dict, self.experiment, target_path)
                    self.targets.append(new_target)
            # TODO test this
            except SSConfigError as e:
                if model_targets: # if targets are listed with no param dict then user messed up
                    print(e)
                    sys.exit(1)
                else:
                    self.logger.info("State created without target, target will have to be created")            
        else:
            self.logger.info("State created without target, target will have to be created")            
            

    def __create_logger(self, log_level):
        import coloredlogs
        logger = logging.getLogger(__name__)
        if log_level == "DEV":
            coloredlogs.install(level=log_level)
        else:
            coloredlogs.install(level=log_level, logger=logger)
        self.logger = logger

    def _get_toml_config(self, path):
        """Searches for configurations in the simulation.toml

           Args
             path (list): a list of strings containing path to config

           Returns
             a configuration value or error is one is not present.
        """
        # Search global configuration file
        try:
            if not self._config:
                raise SSConfigError(self.get_state(),
                                    "Could not find config value for key: "
                                    + ".".join(path))
            else:
                top_level = self.__search_config(path, self._config)
                return top_level
        except SSConfigError as e:
            print(e)
            sys.exit()

    def __search_config(self, value_path, config):
        val_path = value_path.copy()
        # Helper method of _get_config
        if val_path[0] in config.keys():
            if len(val_path) == 1:
                return config[val_path[0]]
            else:
                parent = val_path.pop(0)
                return self.__search_config(val_path, config[parent])
        else:
            raise SSConfigError(self.get_state(),
                                "Could not find config value for key: " + ".".join(value_path))

