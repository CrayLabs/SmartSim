import shutil
import sys
import subprocess
import itertools
import toml

from glob import glob
from os import mkdir, getcwd
from multiprocessing import Pool
from functools import partial

from data_generation.model import NumModel
from data_generation.confwriter import ConfWriter
from error.mpo_error import MpoError
from launcher.Launchers import SlurmLauncher

class Generator():
    """Data generation phase of the MPO pipeline. Holds internal configuration
       data that is created during the data generation stage."""

    def __init__(self, state):
        self.state = state
        self.state.update_state("Data Generation")
        self.low_models = []
        self.high_models = []

    def generate(self):
        try:
            print("MPO Stage: ", self.state.current_state)
            self.create_models()
            self.duplicate_base_configs()
            self.run_models()
        except MpoError as e:
            print(e)

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


    def _create_data_dirs(self):
        try:
            low_dir = getcwd() + "/../low-res-models"
            high_dir = getcwd() + "/../high-res-models"

            # Add paths to data directories to state config
            self.state.set_model_dir("high", high_dir)
            self.state.set_model_dir("low", low_dir)

            mkdir(low_dir)
            mkdir(high_dir)

            return low_dir, high_dir

        except FileExistsError:
            raise MpoError(self.state.get_state(),
                           "Data directories already exist!")

    def duplicate_base_configs(self):

        base_path = self.state.get_config("base_config_path")
        low_dir, high_dir = self._create_data_dirs()

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


    def _sim(self, exe, nodes, model_path, partition="iv24"):
        launcher = SlurmLauncher(def_nodes=nodes, def_partition=partition)
        launcher.validate()
        launcher.get_alloc()
        launcher.run([exe], cwd=model_path)
        launcher.free_alloc()



    def run_models(self):
        exe = self.state.get_config("executable_path")
        nodes_per_low_run = self.state.get_config("low_nodes")
        nodes_per_high_run = self.state.get_config("high_nodes")

        print("   Running low resolution simulations...")
        low_model_dir = self.state.get_model_dir("low")
        for low_model in glob(low_model_dir + "/*"):
            self._sim(exe, nodes_per_low_run, low_model)

        print(" ")
        print("   Running high resolution simulations...")
        high_model_dir = self.state.get_model_dir("high")
        for high_model in glob(high_model_dir + "/*"):
            self._sim(exe, nodes_per_high_run, high_model)
        print(" ")
        print("All simulations complete")
        print("High resolution simulation data:", high_model_dir)
        print("Low resolution simulation data:", low_model_dir)


