import sys
import shutil

from itertools import product
from os import mkdir, getcwd, path
from distutils import dir_util

from ..model import NumModel
from .modelwriter import ModelWriter
from ..error import SmartSimError, SSUnsupportedError, SSConfigError
from ..helpers import get_SSHOME
from ..simModule import SmartSimModule

from .strategies import create_all_permutations, random_permutations, step_values
from ..utils import get_logger
logger = get_logger(__name__)


class Generator(SmartSimModule):
    """A SmartSimModule that configures and generates instances of a model by reading
       and writing model files that have been tagged by the user.

       :param State state: A State instance
       :param list model_files: The model files for the experiment.  Optional
                               if model files are not needed for execution. Argument
                               can be a file, directory, or a list of files
       :param str strategy: The permutation strategy for generating models within targets.
                            Options are "all_perm", "random", "step", or a callable function.
                            defaults to "all_perm"
    """

    def __init__(self, state, model_files=None, strategy="all_perm", **kwargs):
        super().__init__(state, model_files=model_files, **kwargs)
        self.set_state("Data Generation")
        self._writer = ModelWriter()
        self._permutation_strategy = strategy


    def generate(self, **kwargs):
        """Based on the targets and models created by the user,
           configure and generate the model and target instances.

           :param dict kwargs: optional key word arguments passed to permutation strategy.
           :raises: SmartSimError
        """
        try:
            logger.info("SmartSim State: " + self.get_state())
            self.set_strategy(self._permutation_strategy)
            self._create_models(**kwargs)
            self._create_experiment()
            self._configure_models()
        except SmartSimError as e:
            logger.error(e)
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

    def set_strategy(self, permutation_strategy):
        """Load the strategy for generating model configurations based on the
           values of the target parameters.

           all_perm creates all possible permutations of the target parameters as
           individual models. This is the default strategy for the Generator module.

           Calling with a string formatted as "module.function" attempts to use the
           function, `function`, from the module, `module`.

           Calling with a callable function results in that function being used as
           the permutation strategy.

           :param str permutation_strategy: Options are "all_perm", "step", "random",
                                            or a callable function.
           :raises SSUnsupportedError: if strategy is not supported by SmartSim

        """
        if permutation_strategy == "all_perm":
            self._permutation_strategy = create_all_permutations
        elif permutation_strategy == "step":
            self._permutation_strategy = step_values
        elif permutation_strategy == "random":
            self._permutation_strategy = random_permutations
        elif callable(permutation_strategy):
            self._permutation_strategy = permutation_strategy
        else:
            raise SSUnsupportedError("Permutation Strategy given is not supported: " + str(permutation_strategy))


    def _create_models(self, **kwargs):
        """Populates instances of NumModel class for all target models.
           NumModels are created via the function that is set as the
           `_permutation_strategy` attribute.  Users may supply their own
           function (or choose from the available set) via the `set_strategy`
           function.

           By default, the all permutation function ("all_perm") is used.
           This strategy takes all permutations of available configuration
           values and creates a model for each one.

           Returns list of models with configurations to be written
        """

        # collect all parameters, names, and settings
        def read_model_parameters(target):
            param_names = []
            parameters = []
            for name, val in target.params.items():
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
                    raise SmartSimError("Incorrect type for target parameters\n" +
                                        "Must be list, int, or string.")
            return param_names, parameters

        targets = self.get_targets()
        for target in targets:
            # if this call returns empty lists, we shouldn't continue.
            # This is useful for empty targets where the user makes models.
            names, values = read_model_parameters(target)
            if (len(names) != 0 and len(values) != 0):
                all_configs = self._permutation_strategy(names, values, **kwargs)
                for i, conf in enumerate(all_configs):
                    model_name = "_".join((target.name, str(i)))
                    m = NumModel(model_name, conf, i)
                    target.add_model(m)

    def _create_experiment(self):
        """Creates the directory structure for the simulations"""
        #TODO: add argument to override target creation
        exp_path = self.get_experiment_path()

        # ok to have already created an experiment
        if not path.isdir(exp_path):
            mkdir(exp_path)
        else:
            logger.error("Working in previously created experiment")

        # not ok to have already generated the target.
        targets = self.get_targets()
        for target in targets:
            target_dir = path.join(exp_path, target.name)
            if not path.isdir(target_dir):
                mkdir(target_dir)
            else:
                raise SmartSimError("Models for an experiment by this name have already been generated!")

    def _configure_models(self):
        """Duplicate the base configurations of target models"""

        listed_configs = self.get_config("model_files")
        exp_path = self.get_experiment_path()
        targets = self.get_targets()

        for target in targets:
            target_models = target.models

            # Make target model directories
            for name, model in target_models.items():
                dst = path.join(exp_path, target.name, name)
                mkdir(dst)
                model.path = (dst)

                if listed_configs:
                    if not isinstance(listed_configs, list):
                        listed_configs = [listed_configs]
                    for config in listed_configs:
                        dst_path = path.join(dst, path.basename(config))
                        config_path = path.join(getcwd(), config)
                        if path.isdir(config_path):
                            dir_util.copy_tree(config_path, dst)
                        else:
                            shutil.copyfile(config_path, dst_path)

                    # write in changes to configurations
                    self._writer.write(model)