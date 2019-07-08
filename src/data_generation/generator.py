import subprocess
import itertools
import logging

from glob import glob
from os import mkdir, getcwd
from multiprocessing import Pool
from functools import partial

from data_generation.model import NumModel
from error.ss_error import SmartSimError, SSUnsupportedError
from launcher.Launchers import SlurmLauncher
from writers import *

class Generator():
    """Data generation phase of the Smart Sim pipeline. Holds internal configuration
       data that is created during the data generation stage.

       Args
         state  (State instance): The state of the library
    """

    def __init__(self, state):
        self.state = state
        self.state.update_state("Data Generation")
        self.low_models = []
        self.high_models = []

    def generate(self):
        """Generate model runs according to the main configuration file"""
        try:
            logging.info("Smart Sim Stage: %s", self.state.current_state)
            self._create_models()
            self._duplicate_base_configs()
            #self._run_models()
        except SmartSimError as e:
            print(e)

    def _create_models(self):
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
        """Create data directories to house simulation data"""
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
            raise SmartSimError(self.state.get_state(),
                           "Data directories already exist!")

    def _duplicate_base_configs(self):
        """Duplicate the base configurations of the numerical model"""

        base_path = self.state.get_config("base_config_path")
        low_dir, high_dir = self._create_data_dirs()

        for low_run in self.low_models:
            dup_path = low_dir + "/"+ low_run.name
            create_low_dirs = subprocess.Popen("cp -r " + base_path +
                                                " " + dup_path,
                                               shell=True)
            create_low_dirs.wait()
            self._write_parameters(dup_path, low_run.param_dict)
            self._write_model_configs(dup_path, low_run.settings)


        for high_run in self.high_models:
            dup_path = high_dir + "/" + high_run.name
            create_high_dirs = subprocess.Popen("cp -r " + base_path +
                                                " " + dup_path,
                                                shell=True)
            create_high_dirs.wait()
            self._write_parameters(dup_path, high_run.param_dict)
            self._write_model_configs(dup_path, high_run.settings)


    def _get_config_writer(self):
        """Find and return the configuration writer for this model"""

        model_name = self.state.get_config("model_name")
        if model_name == "MOM6":
            writer = mom6_writer.MOM6Writer()
            return writer
        else:
            raise SSUnsupportedError("Model not supported yet")


    def _write_parameters(self, base_conf_path, param_dict):
        """Write the model instance specific parameters from
           models createed in _create_models"""

        param_info = self.state.get_config("parameter_info")
        filename = param_info["filename"]
        filetype = param_info["filetype"]
        full_path = base_conf_path + "/" + filename

        conf_writer = self._get_config_writer()
        conf_writer.write_config(param_dict, full_path, filetype)


    def _write_model_configs(self, base_conf_path, config_dict):
        """Write the model instance specifc run configurations"""

        # TODO handle errors for when this info isnt present
        conf_writer = self._get_config_writer()
        for name, config_info in config_dict.items():
            filename = config_info["filename"]
            filetype = config_info["filetype"]
            value = config_info["value"]
            full_path = base_conf_path + "/" + filename
            conf_writer.write_config({name: value}, full_path, filetype)


    def _sim(self, exe, nodes, model_path, partition="iv24"):
        """Simulate a model that has been configured by the generator
           Currently uses the slurm launcher

           Args
              exe        (str): path to the compiled numerical model executable
              nodes      (int): number of nodes to run on for this model
              model_path (str): path to dir that houses model configurations
              partition  (str): type of proc to run on (optional)
        """
        launcher = SlurmLauncher(def_nodes=nodes, def_partition=partition)
        launcher.validate()
        launcher.get_alloc()
        launcher.run([exe], cwd=model_path)
        launcher.free_alloc()



    def _run_models(self):
        """Run all models that have been configured by the generator"""

        exe = self.state.get_config("executable_path")
        nodes_per_low_run = self.state.get_config("low_nodes")
        nodes_per_high_run = self.state.get_config("high_nodes")

        logging.info("   Running low resolution simulations...")
        low_model_dir = self.state.get_model_dir("low")
        for low_model in glob(low_model_dir + "/*"):
            self._sim(exe, nodes_per_low_run, low_model)

        logging.info("   Running high resolution simulations...")
        high_model_dir = self.state.get_model_dir("high")
        for high_model in glob(high_model_dir + "/*"):
            self._sim(exe, nodes_per_high_run, high_model)
        logging.info("All simulations complete")
        logging.info("High resolution simulation data: %s", high_model_dir)
        logging.info("Low resolution simulation data: %s", low_model_dir)


