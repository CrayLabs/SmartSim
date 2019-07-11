import subprocess
import itertools
import logging
import sys

from os import mkdir, getcwd

from generation.model import NumModel
from error.errors import SmartSimError, SSUnsupportedError
from launcher.Launchers import SlurmLauncher
from writers import *
from helpers import get_SSHOME
from ssModule import SSModule

class Generator(SSModule):
    """Data generation phase of the Smart Sim pipeline. Holds internal configuration
       data that is created during the data generation stage.

       Args
         state  (State instance): The state of the library
    """

    def __init__(self, state, local_config="generate.toml"):
        super().__init__(state, local_config)
        self.state.update_state("Data Generation")
        self.model_list = []

    def generate(self):
        """Generate model runs according to the main configuration file"""
        try:
            logging.info("SmartSim Stage: %s", self.state.get_state())
            self._create_models()
            self._create_data_dirs()
            self._duplicate_and_configure()
        except SmartSimError as e:
            print(e)
            sys.exit()

    def _create_models(self):
        """Populates instances of NumModel class for all target models.
           Targets are retieved from state and permuted into all possible
           model configurations.

           Returns: List of models with configurations to be written
        """

        # collect all parameters, names, and settings
        def read_model_parameters(target):
            target_params = self.get_config([target])
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
        for target in self.targets:
            names, values, settings = read_model_parameters(target)
            all_configs = create_all_permutations(names, values)
            for conf in all_configs:
                m = NumModel(target, conf, settings)
                self.model_list.append(m)


    def _create_data_dirs(self):
        """Create data directories to house simulation data"""

        try:
            for target in self.targets:
                target_dir = get_SSHOME() + target
                mkdir(target_dir)

        except FileExistsError:
            raise SmartSimError(self.state.get_state(),
                           "Data directories already exist!")

    def _duplicate_and_configure(self):
        """Duplicate the base configurations of the numerical model"""

        base_path = self.get_config(["model","base_config_path"])

        for target in self.targets:
            for model in self.model_list:
                name = model.name
                if name.startswith(target):
                    dup_path = get_SSHOME() + "/".join([target, name])
                    create_target_dir = subprocess.Popen("cp -r " + base_path +
                                                         " " + dup_path,
                                                         shell=True)
                    create_target_dir.wait()
                    self._write_parameters(dup_path, model)

    def _get_config_writer(self):
        """Find and return the configuration writer for this model"""

        model_name = self.get_config(["model","name"])
        if model_name == "MOM6":
            writer = mom6_writer.MOM6Writer()
            return writer
        else:
            raise SSUnsupportedError("Model not supported yet")


    def _write_parameters(self, base_conf_path, model):
        """Write the model instance specifc run configurations

           Args
             base_conf_path (str): filepath to duplicated model
             model (Model): the Model instance to write parameters for

        """
        conf_writer = self._get_config_writer()
        for name, param_info in model.param_settings.items():
            filename = param_info["filename"]
            filetype = param_info["filetype"]
            value = model.param_dict[name]
            full_path = "/".join((base_conf_path, filename))
            conf_writer.write_config({name: value}, full_path, filetype)

