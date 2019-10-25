import sys
import shutil

from itertools import product
from os import mkdir, getcwd, path
from distutils import dir_util

from ..model import NumModel
from .modelwriter import ModelWriter
from ..error import SmartSimError, SSUnsupportedError
from ..helpers import get_SSHOME
from ..simModule import SmartSimModule


class Generator(SmartSimModule):
    """A SmartSimModule that configures and generates instances of a model by reading
       and writing model files that have been tagged by the user.

       :param State state: A State instance
    """

    def __init__(self, state, log_level="DEV", **kwargs):
        super().__init__(state, __name__, log_level=log_level, **kwargs)
        self.set_state("Data Generation")
        self._writer = ModelWriter()


    def generate(self):
        """Based on the targets and models created by the user through the python
           or TOML interface, configure and generate the model instances.

           :raises: SmartSimError
        """
        try:
            self.logger.info("SmartSim State: " + self.get_state())
            self._create_models()
            self._create_experiment()
            self._configure_models()
        except SmartSimError as e:
            self.logger.error(e)
            raise

    def set_tag(self, tag, regex=None):
        """Set a tag or a regular expression for the generator to look for when
           configuring new models.

           For example, a tag might be ``;`` where the expression being replaced
           in the model configuration file would look like ``;expression;``

           A full regular expression might tag specific model configurations such
           that the configuration files don't need to be tagged manually.

           :param str tag: A string of characters that signify an string to be changed.
                           Defaults to ``;``
           :param str regex: a regular expression that model files are tagged with

        """

        self._writer._set_tag(tag, regex)

    def select_strategy(self, strategy):
        """Select the strategy for generating model configurations based on the
           values of the target parameters.

           all_perm creates all possible permutations of the target parameters as
           individual models. This is the default strategy for the Generator module

           :param str strategy: Options are "all_perm"

        """
        raise NotImplementedError



    def _create_models(self):
        """Populates instances of NumModel class for all target models.
           NumModels are created via a strategy of which there is only
           one implemented: all permutations.

           This strategy takes all permutations of available configuration
           values and creates a model for each one.

           Returns list of models with configurations to be written
        """

        # collect all parameters, names, and settings
        def read_model_parameters(target):
            target_params = target.get_target_params()
            param_names = []
            parameters = []
            for name, val in target_params.items():
                param_names.append(name)

                # if it came from a simulation.toml
                if isinstance(val, dict):
                    if isinstance(val["value"], list):
                        parameters.append(val["value"])
                    else:
                        parameters.append([val["value"]])

                # if the user called added a target programmatically
                elif isinstance(val, list):
                    parameters.append(val)
                elif isinstance(val, str) or isinstance(val, int):
                    parameters.append([val])
                else:
                    # TODO improve this error message
                    raise SmartSimError(self.get_state(),
                                        "Incorrect type for target parameters\n" +
                                        "Must be list, int, or string.")
            return param_names, parameters


        # init model classes to hold parameter information
        targets = self.get_targets()
        for target in targets:
            names, values = read_model_parameters(target)
            # TODO Allow for different strategies to be used
            all_configs = self._create_all_permutations(names, values)
            for i, conf in enumerate(all_configs):
                model_name = "_".join((target.name, str(i)))
                m = NumModel(model_name, conf, i)
                target.add_model(m)

    def _create_experiment(self):
        """Creates the directory structure for the simulations"""
        exp_path = self.get_experiment_path()

        # ok to have already created an experiment
        try:
            mkdir(exp_path)
        except FileExistsError:
            self.logger.error("Working in previously created experiment")

        # not ok to have already generated the target.
        try:
            targets = self.get_targets()
            for target in targets:
                target_dir = path.join(exp_path, target.name)
                mkdir(target_dir)

        except FileExistsError:
            raise SmartSimError(self.get_state(),
                        "Models for an experiment by this name have already been generated!")




    def _configure_models(self):
        """Duplicate the base configurations of target models"""

        listed_configs = self.get_config(["model", "model_files"])
        exp_path = self.get_experiment_path()
        targets = self.get_targets()

        for target in targets:
            target_models = target.get_models()

            # Make target model directories
            for name, model in target_models.items():
                dst = path.join(exp_path, target.name, name)
                mkdir(dst)
                model.set_path(dst)

                if not isinstance(listed_configs, list):
                    listed_configs = [listed_configs]
                for config in listed_configs:
                    dst_path = path.join(dst, path.basename(config))
                    config_path = path.join(get_SSHOME(), config)
                    if path.isdir(config_path):
                        dir_util.copy_tree(config_path, dst)
                    else:
                        shutil.copyfile(config_path, dst_path)

                # write in changes to configurations
                self._writer.write(model)



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
