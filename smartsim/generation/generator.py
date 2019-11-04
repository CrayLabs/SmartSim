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


class Generator(SmartSimModule):
    """A SmartSimModule that configures and generates instances of a model by reading
       and writing model files that have been tagged by the user.

       :param State state: A State instance
    """

    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self.set_state("Data Generation")
        self._writer = ModelWriter()
        self._permutation_strategy = None


    def generate(self, **kwargs):
        """Based on the targets and models created by the user through the python
           or TOML interface, configure and generate the model instances.

           :raises: SmartSimError
        """
        try:
            self.log("SmartSim State: " + self.get_state())
            if self._permutation_strategy == None:
                self._set_strategy_from_config()
            self._create_models(**kwargs)
            self._create_experiment()
            self._configure_models()
        except SmartSimError as e:
            self.log(e, level="error")
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
                                            "module.function", or a callable function.


        """
        if callable(permutation_strategy):
            self._permutation_strategy = permutation_strategy
        else:
            self._set_strategy_from_string(permutation_strategy)

    def _set_strategy_from_config(self):
        """Load the strategy for generating model configurations from the supplied
        configuration; if a user has specified anything, it's passed on to
        the _set_strategy_from_string function for parsing.  Otherwise,
        _set_strategy_from_string uses a default value ("all_perm")
        """
        # first, check to see what strategy we've selected in the config, if we've
        # bothered to select one.
        try:
            permutation_strategy = self.get_config(["model", "permutation"])
            self._set_strategy_from_string(permutation_strategy)
        except SSConfigError:
            # if we couldn't find the field, choose a reasonable default (all)
            self._set_strategy_from_string()       

    def _set_strategy_from_string(self, permutation_strategy="all_perm"):
        """Sets the strategy for generating model configurations based on the
           supplied string, `permutation_strategy`.  `permutation_strategy` can
           be a string corresponding to an internal function name (for the built-in
           strategies), or of the form `module.function`, where module is importable
           and has the function `function` available on it.

           :param str permutation_strategy: can be "all_perm", "step", or "random" for
           the built-in functions, or "module.function".

        """
        if permutation_strategy == "all_perm":
            self._permutation_strategy = self._create_all_permutations
        elif permutation_strategy == "step":
            self._permutation_strategy = self._step_values
        elif permutation_strategy == "random":
            self._permutation_strategy = self._random_permutations
        else:
            # return a function that the user thinks is appropriate.  Assume module.function
            import importlib
            try:
                mod_string, func_string = permutation_strategy.split(".")
            except:
                raise SmartSimError(self.current_state,
                                    "Following string cannot be evaluated to a module.function: ", permutation_strategy)
            try:
                mod = importlib.import_module(mod_string)
            except:
                raise
            try:
                func = getattr(mod, func_string)
            except:
                raise
            if callable(func):
                self._permutation_strategy = func
            else:
                raise SmartSimError(self.current_state,
                                    "Supplied attribute is not a function: ", func)



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
            # if this call returns empty lists, we shouldn't continue.
            # This is useful for empty targets where the user makes models.
            names, values = read_model_parameters(target)
            if (len(names) != 0 and len(values) != 0):
                # TODO Allow for different strategies to be used
                all_configs = self._permutation_strategy(names, values, **kwargs)
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
            self.log("Working in previously created experiment")

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
    def _step_values(param_names, param_values):
        permutations = []
        for p in zip(*param_values):
            permutations.append(dict(zip(param_names, p)))
        return permutations

    @staticmethod
    def _random_permutations(param_names, param_values, n_models):
        # a basic, random example.  Unknown performance.
        import random
        # first, check if we've requested more values than possible.
        perms = list(product(*param_values))
        if n_models >= len(perms):
            # This is literally just _create_all_permutations
                all_permutations = []
                for p in perms:
                    temp_model = dict(zip(param_names, p))
                    all_permutations.append(temp_model)
                return all_permutations
        else:
            permutations = []
            permutation_strings = set()
            while len(permutations) < n_models:
                model_dict = dict(zip(param_names, map(lambda x: x[random.randint(0,len(x)-1)], param_values)))
                if str(model_dict) not in permutation_strings:
                    permutation_strings.add(str(model_dict))
                    permutations.append(model_dict)
            return permutations

    @staticmethod
    def _one_per_change():
        raise NotImplementedError

    @staticmethod
    def _hpo():
        raise NotImplementedError
