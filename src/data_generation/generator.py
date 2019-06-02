import shutil
import glob
import sys
import subprocess
import itertools
import toml


from os import mkdir, getcwd
from data_generation.model import NumModel
from data_generation.confwriter import ConfWriter
from data_generation.runner import ModelRunner

class Generator():
    """Data generation phase of the MPO pipeline. Holds internal configuration
       data that is created during the data generation stage."""

    def __init__(self, state):
        self.state = state
        self.state.update_state("Data Generation")
        self.low_models = []
        self.high_models = []

    def generate(self):
        print("MPO Stage: ", self.state.current_state)
        self.create_models()
        self.duplicate_base_configs()
        print("     Writing configurations...")
        self.run_models()
        print("     Running simulations...")

    def create_models(self):
        """Populates instances of NumModel class for low and high resolution.
           obtains parameter permutations from state.

           Returns: list of high and low resolution Model objects for data
                    generation
        """

        param_dict = self.state.get_config("parameters")
        permutations = list(itertools.product(*param_dict.values()))
        params = list(param_dict.keys())

        for p in permutations:
            model_params = dict(zip(params, list(p)))
            settings = self.state.get_config("low")
            m = NumModel(model_params, settings)
            self.low_models.append(m)

        for p in permutations:
            model_params = dict(zip(params, list(p)))
            settings = self.state.get_config("high")
            m = NumModel(model_params, settings)
            self.high_models.append(m)



    def duplicate_base_configs(self):
        # TODO catch if base config path is not found
        base_path = self.state.get_config("MPO_settings")["base_config_path"]

        # Make data directories
        # TODO catch if data dirs exist already
        low_dir = getcwd() + "/../low-res-models"
        high_dir = getcwd() + "/../high-res-models"
        mkdir(low_dir)
        mkdir(high_dir)

        # Add paths to data directories to state config
        self.state.set_model_dir("high", high_dir)
        self.state.set_model_dir("low", low_dir)

        for low_run in self.low_models:
            dup_path = low_dir + "/"+ low_run.name
            create_low_dirs = subprocess.Popen("cp -r " + base_path +
                                                " " + dup_path,
                                               shell=True)
            create_low_dirs.wait()
            self.write_parameters(dup_path, low_run.param_dict)
            self.write_model_configs(dup_path, low_run.settings)


        for high_run in self.high_models:
            dup_path = high_dir + "/" + high_run.name
            create_high_dirs = subprocess.Popen("cp -r " + base_path +
                                                " " + dup_path,
                                                shell=True)
            create_high_dirs.wait()
            self.write_parameters(dup_path, high_run.param_dict)
            self.write_model_configs(dup_path, high_run.settings)

    def write_parameters(self, base_conf_path, param_dict):
        param_info = self.state.get_config("parameter_info")
        filename = param_info["filename"]
        filetype = param_info["filetype"]
        full_path = base_conf_path + "/" + filename

        conf_writer = ConfWriter()
        conf_writer.write_config(param_dict, full_path, filetype)



    def write_model_configs(self, base_conf_path, config_dict):
        # TODO handle errors for when this info isnt present
        conf_writer = ConfWriter()
        for name, config_info in config_dict.items():
            filename = config_info["filename"]
            filetype = config_info["filetype"]
            value = config_info["value"]
            full_path = base_conf_path + "/" + filename
            conf_writer.write_config({name: value}, full_path, filetype)


    def run_models(self):
        exe = self.state.get_config("MPO_settings")["executable_path"]
        low_node_count = self.state.get_config("MPO_settings")["low_nodes"]
        high_node_count = self.state.get_config("MPO_settings")["high_nodes"]
        procs_per_node = self.state.get_config("MPO_settings")["procs_per_node"]

        # run low resolution models
        low_model_dir = self.state.get_model_dir("low")
        runner = ModelRunner(exe, low_node_count, procs_per_node)
        runner.run_all_models(low_model_dir)

        # run high resolution models
        high_model_dir = self.state.get_model_dir("high")
        runner = ModelRunner(exe, high_node_count, procs_per_node)
        runner.run_all_models(high_model_dir)
