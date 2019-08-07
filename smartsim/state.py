import logging
from .helpers import read_config
from .error import SmartSimError, SSConfigError


class State:
    """Holds the state of the entire SS pipeline and the configurations
       necessary to run each stage of the pipeline"""

    def __init__(self, config="simulation.toml", log_level="DEV"):
        self.current_state = "Initializing"
        self.config = read_config(config)
        self.logger = self.create_logger2(log_level)

    def get_state(self):
        return self.current_state

    def update_state(self, new_state):
        self.current_state = new_state


    def create_logger2(self, log_level):
        import coloredlogs
        logger = logging.getLogger(__name__)
        if log_level == "DEV":
            coloredlogs.install(level=log_level)
        else:
            coloredlogs.install(level=log_level, logger=logger)
        return logger

