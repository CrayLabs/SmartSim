import logging

from helpers import read_config
from error.errors import SmartSimError, SSConfigError


class State:
    """Holds the state of the entire SS pipeline and the configurations
       necessary to run each stage of the pipeline"""

    def __init__(self, config="simulation.toml"):
        self.current_state = "Initializing"
        self.config = read_config(config)
        logging.info("SmartSim State: %s", self.current_state)

    def get_state(self):
        return self.current_state

    def update_state(self, new_state):
        self.current_state = new_state


