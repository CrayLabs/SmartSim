

from helpers import read_config

class State:
    """Holds the state of the entire MPO pipeline and the configurations
       necessary to run each stage of the pipeline"""

    def __init__(self):
        self.config = read_config()
        self.current_state = "Initializing"
        print("MPO State: ", self.current_state)
        self.high_model_dir = None
        self.low_model_dir = None


    def get_state(self):
        return self.current_state

    def update_state(self, new_state):
        self.current_state = new_state

    def get_config(self, table_name):
        return self.config[table_name]

    def set_model_dir(self, res, path):
        if res == "high":
            self.high_model_dir = path
        else:
            self.low_model_dir = path

    def get_model_dir(self, res):
        if res == "high":
            return self.high_model_dir
        return self.low_model_dir

