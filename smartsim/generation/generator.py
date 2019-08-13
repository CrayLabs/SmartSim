import sys
import shutil

from itertools import product
from os import mkdir, getcwd
from os.path import isdir, basename
from distutils import dir_util

from .model import NumModel
from .modelwriter import ModelWriter
from ..error import SmartSimError, SSUnsupportedError
from ..helpers import get_SSHOME
from ..ssModule import SSModule

"""
Generation

 - models are created based on the content of the simulation.toml
   that will be populated as a result of the interface or manual
   creation.
 - models are created with the following tree for and example 1 target with
   two resulting models
   - lammps_atm/              (experiment name)
     └── atm                  (target name)
         ├── atm_ld           (model name)
         │    └── in.atm
         └── atm_ls           (model name)
              └── in.atm

A configuration file for this generation could look like the following when generated
with the all permutations strategy.

```toml
[model]
name = "lammps"
experiment = "lammps_atm"
configs = ["in.atm"]

[execute]
targets = ["atm"]

[atm]
  [atm.lj]              # lj is the previous value marked in "in.atm" (e.g. ;lj;)
  value = ["ls", "ld"]
```

"""


class Generator(SSModule):
    """Data generation phase of the Smart Sim pipeline. Holds internal configuration
       data that is created during the data generation stage.

       Args
         state  (State instance): The state of the library
    """

    def __init__(self, state):
        super().__init__(state)
        self.state.update_state("Data Generation")
        self._writer = ModelWriter()
        self._models = {}

    def generate(self):
        """Generate model runs according to the main configuration file
           Note that this only generates the necessary files and structure
           to be able to run all models in parallel, it does not actually
           run any models."""
        try:
            self.log("SmartSim Stage: " + self.state.get_state())
            self._create_models()
            self._create_experiment()
            self._configure_models()
        except SmartSimError as e:
            print(e)
            sys.exit()

    def _create_models(self):
        """Populates instances of NumModel class for all target models.
           NumModels are created via a strategy of which there is only
           one implemented: all permutations.

           This strategy takes all permutations of available configuration
           values and creates a model for each one.

           Returns: List of models with configurations to be written
        """

        # collect all parameters, names, and settings
        def read_model_parameters(target):
            target_params = self._get_config([target])
            param_names = []
            parameters = []
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
            all_configs = self._create_all_permutations(names, values)
            for conf in all_configs:
                m = NumModel(target, conf)
                if target not in self._models.keys():
                    self._models[target] = [m]
                else:
                    self._models[target].append(m)

    def _create_experiment(self):
        """Creates the directory structure for the simluations"""
        base_path = "".join((get_SSHOME(), self._get_config(["model","name"])))
        exp_name = self._get_config(["model", "experiment"])
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

        base_path = "".join((get_SSHOME(), self._get_config(["model","name"])))
        listed_configs = self._get_config(["model", "configs"])

        for target, target_models in self._models.items():

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
                self._writer.write(model, dst)



######################
### run strategies ###
######################

    # create permutations of all parameters
    # single model if parameters only have one value
    @staticmethod
    def _create_all_permutations(param_names, param_values):
        perms = list(product(*param_values))
        all_permutations = []
        for p in perms:
            temp_model = dict(zip(param_names, p))
            all_permutations.append(temp_model)
        return all_permutations

    @staticmethod
    def _one_per_change():
        raise NotImplementedError

    @staticmethod
    def _hpo():
        raise NotImplementedError

