import subprocess
import itertools
import logging
import sys

from os import mkdir, getcwd
from shutil import copyfile

from generation.model import NumModel
from generation.modelwriter import ModelWriter
from error.errors import SmartSimError, SSUnsupportedError
from launcher.Launchers import SlurmLauncher
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
                if isinstance(val["values"], list):
                    parameters.append(val["values"])
                else:
                    parameters.append([val["values"]])
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
        base_path = "/".join((get_SSHOME(), self.get_config(["model","name"])))
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

    def _init_model_writer(self, target_configs):
        writer = ModelWriter(target_configs)
        self.writer = writer


    def _configure_models(self):
        """Duplicate the base configurations of target models"""

        # init the model writer class
        target_configs = self.get_config(["model", "configs"])
        self._init_model_writer(target_configs)

        # copy base configuration files to new model dir within target dir
        base_path = get_SSHOME() + self.get_config(["model", "name"])
        for target, target_models in self.models.items():
            for model in target_models:
                dst = "/".join((self.exp_path, target, model.name))
                mkdir(dst)
                for conf in target_configs:
                    # TODO make this copy all files and directories
                    copyfile("/".join((base_path, conf)), "/".join((dst, conf)))

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

