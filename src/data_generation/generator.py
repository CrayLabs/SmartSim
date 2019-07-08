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
        self.model_list = []

    def generate(self):
        """Generate model runs according to the main configuration file"""
        try:
            logging.info("SmartSim Stage: %s", self.state.current_state)
            self._create_models()
            self._create_data_dirs()
            self._duplicate_and_configure()

        except SmartSimError as e:
            print(e)

    def _create_models(self):
        """Populates instances of NumModel class for all target models.
           Targets are retieved from state and permuted into all possible
           model configurations.

           Returns: List of models with configurations to be written
        """

        # collect all parameters, names, and settings
        def read_model_parameters(target):
            target_params = self.state.get_config(target)
            param_names = []
            parameters = []
            param_settings = {}
            for name, val in target_params.items():
                param_names.append(name)
                param_settings[name] = val
                if isinstance(val["value"], list):
                    parameters.append(val["value"])
                else:
                    parameters.append([val["value"]])
            return param_names, parameters, param_settings

        # create permutations of all parameters
        # single model if parameters only have one value
        def create_all_permutations(param_names, param_values):
            perms = list(itertools.product(*param_values))
            all_permutations = []
            for p in perms:
                temp_model = dict(zip(param_names, p))
                all_permutations.append(temp_model)
            return all_permutations

        # init model classes to hold parameter information
        targets = self.state.get_config("targets")
        for target in targets:
            names, values, settings = read_model_parameters(target)
            all_configs = create_all_permutations(names, values)
            for conf in all_configs:
                m = NumModel(target, conf, settings)
                self.model_list.append(m)


    def _create_data_dirs(self):
        """Create data directories to house simulation data"""
        # TODO Let user specify where to put data
        # or create environment variables
        targets = self.state.get_config("targets")
        try:
            for target in targets:
                target_dir = "/".join((getcwd(), "..", target))
                mkdir(target_dir)

        except FileExistsError:
            raise SmartSimError(self.state.get_state(),
                           "Data directories already exist!")

    def _duplicate_and_configure(self):
        """Duplicate the base configurations of the numerical model"""

        base_path = self.state.get_config("base_config_path")
        targets = self.state.get_config("targets")

        for target in targets:
            for model in self.model_list:
                name = model.name
                if name.startswith(target):
                    dup_path = "/".join(("..", target, name))
                    create_target_dir = subprocess.Popen("cp -r " + base_path +
                                                         " " + dup_path,
                                                         shell=True)
                    create_target_dir.wait()
                    self._write_parameters(dup_path, model)

    def _get_config_writer(self):
        """Find and return the configuration writer for this model"""

        model_name = self.state.get_config("name")
        if model_name == "MOM6":
            writer = mom6_writer.MOM6Writer()
            return writer
        else:
            raise SSUnsupportedError("Model not supported yet")


    def _write_parameters(self, base_conf_path, model):
        """Write the model instance specifc run configurations"""
        conf_writer = self._get_config_writer()
        for name, param_info in model.param_settings.items():
            filename = param_info["filename"]
            filetype = param_info["filetype"]
            value = model.param_dict[name]
            full_path = "/".join((base_conf_path, filename))
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

