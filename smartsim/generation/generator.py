import subprocess
import itertools
import logging
import sys
import shutil

from os import mkdir, getcwd
from os.path import isdir, basename
from distutils import dir_util
from glob import glob

from generation.model import NumModel
from generation.modelwriter import ModelWriter
from error.errors import SmartSimError, SSUnsupportedError
from helpers import get_SSHOME
from ssModule import SSModule



class Generator(SSModule):
    """Data generation phase of the Smart Sim pipeline. Holds internal configuration
       data that is created during the data generation stage.

       Args
         state  (State instance): The state of the library
    """

    def __init__(self, state):
        super().__init__(state)
        self.state.update_state("Data Generation")
        self.writer = ModelWriter()
        self.models = {}

    def generate(self):
        """Generate model runs according to the main configuration file"""
        try:
            logging.info("SmartSim Stage: %s", self.state.get_state())
            self._create_models()
            self._create_experiment()
            self._configure_models()
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
                if isinstance(val["value"], list):
                    parameters.append(val["value"])
                else:
                    parameters.append([val["value"]])
            return param_names, parameters


        # init model classes to hold parameter information
        for target in self.targets:
            names, values = read_model_parameters(target)
            all_configs = self.create_all_permutations(names, values)
            for conf in all_configs:
                m = NumModel(target, conf)
                if target not in self.models.keys():
                    self.models[target] = [m]
                else:
                    self.models[target].append(m)

    def _create_experiment(self):
        """Creates the directory stucture for the simluations"""
        base_path = "".join((get_SSHOME(), self.get_config(["model","name"])))
        exp_name = self.get_config(["model", "experiment_name"])
        exp_dir_path = "/".join((base_path, exp_name))
        self.exp_path = exp_dir_path

        try:
            mkdir(exp_dir_path)
            for target in self.targets:
                target_dir = "/".join((exp_dir_path, target))
                mkdir(target_dir)

        except FileExistsError:
            raise SmartSimError(self.state.get_state(),
                           "Data directories already exist!")



    def _configure_models(self):
        """Duplicate the base configurations of target models"""

        base_path = "".join((get_SSHOME(), self.get_config(["model","name"])))
        listed_configs = self.get_config(["model", "configs"])

        for target, target_models in self.models.items():

            # Make target model directories
            for model in target_models:
                dst = "/".join((self.exp_path, target, model.name))
                mkdir(dst)

                # copy over model base configurations
                for config in listed_configs:
                    dst_path = "/".join((dst, config))
                    config_path = "/".join((base_path, config))
                    if isdir(config_path):
                        dir_util.copy_tree(config_path, dst)
                    else:
                        shutil.copyfile(config_path, dst_path)

                # write in changes to configurations
                self.writer.write(model, dst)



######################
### run strategies ###
######################

    # create permutations of all parameters
    # single model if parameters only have one value
    @staticmethod
    def create_all_permutations(param_names, param_values):
        perms = list(itertools.product(*param_values))
        all_permutations = []
        for p in perms:
            temp_model = dict(zip(param_names, p))
            all_permutations.append(temp_model)
        return all_permutations

    @staticmethod
    def one_per_change():
        raise NotImplementedError

    @staticmethod
    def hpo():
        raise NotImplementedError

