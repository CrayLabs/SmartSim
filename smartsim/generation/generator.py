import sys
import shutil

from itertools import product
from os import mkdir, getcwd, path
from distutils import dir_util

from ..model import NumModel
from ..ensemble import Ensemble
from .modelwriter import ModelWriter
from ..orchestrator import Orchestrator
from ..smartSimNode import SmartSimNode
from ..error import SmartSimError, SSUnsupportedError, SSConfigError

from .strategies import create_all_permutations, random_permutations, step_values
from ..utils import get_logger
logger = get_logger(__name__)


class Generator():
    """Class used by Experiment to generate the file structure for an experiment. The
       Generator is responsible for reading, configuring, and writing model instances
       within an ensemble created by an Experiment.
    """
    def __init__(self):
        self._writer = ModelWriter()
        self.set_strategy("all_perm")

    def generate_experiment(self,
                            exp_path,
                            ensembles=[],
                            nodes=[],
                            orchestrator=None,
                            model_files=[],
                            node_files=[],
                            **kwargs):
        """Generate the file structure for a SmartSim experiment. This includes the writing
           and configuring of input files for a model. Ensembles created with a 'params' argument
           will be expanded into multiple models based on a generation strategy. Model input files
           are specified with the model_files argument. All files and directories listed as strings
           in a list will be copied to each model within an ensemble. Every model file is read,
           checked for input variables to configure, and written. Input variables to configure
           are specified with a tag within the input file itself. The default tag is surronding
           an input value with semicolons. e.g. THERMO=;90;

           Files for SmartSimNodes can also be included to be copied into node directories but
           are never read nor written. All node_files will be copied into directories named after
           the name of the SmartSimNode within the experiment.

           :param str exp_path: Path to the directory, usually an experiment directory, where the
                                directory structure will be created.
           :param ensembles:  ensembles created by an Experiment to be generated
           :type ensembles: List of Ensemble objects
           :param nodes: nodes created by an experiment to be generated
           :type nodes: list of SmartSimNodes
           :param orchestrator: orchestrator object for file generation
           :type orchestrator: Orchestrator object
           :param model_files: files or directories to be read, configured and written for every
                               model instance within an ensemble
           :type model_files: a list of string paths
           :param node_files: files or directories to be included in each node directory
           :type node_files: a list of string paths
           :param dict kwargs: optional key word arguments passed to generation strategy.
           :raises: SmartSimError
        """
        if isinstance(ensembles, Ensemble):
            ensembles = [ensembles]
        if isinstance(nodes, SmartSimNode):
            nodes = [nodes]
        if isinstance(model_files, str):
            model_files = [model_files]
        if isinstance(node_files, str):
            node_files = [node_files]
        if orchestrator and not isinstance(orchestrator, Orchestrator):
            raise SmartSimError(
                f"Argument given for orchestrator is of type {type(orchestrator)}, not Orchestrator"
            )
        self._generate_ensembles(ensembles, **kwargs)
        self._create_experiment_dir(exp_path)
        self._create_orchestrator_dir(exp_path, orchestrator)
        self._create_nodes(exp_path, nodes, node_files)
        self._create_ensembles(exp_path, ensembles)
        self._configure_models(ensembles, exp_path, model_files)


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
           values of the ensemble parameters.

           all_perm creates all possible permutations of the ensemble parameters as
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
            raise SSUnsupportedError(
                "Permutation Strategy given is not supported: " +
                str(permutation_strategy))

    def _generate_ensembles(self, ensembles, **kwargs):
        """Populates instances of NumModel class for all ensemble models.
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
        def read_model_parameters(ensemble):
            param_names = []
            parameters = []
            for name, val in ensemble.params.items():
                param_names.append(name)

                # if the user called added a ensemble programmatically
                if isinstance(val, list):
                    parameters.append(val)
                elif isinstance(val, str) or isinstance(val, int):
                    parameters.append([val])
                else:
                    # TODO improve this error message
                    raise SmartSimError(
                        "Incorrect type for ensemble parameters\n" +
                        "Must be list, int, or string.")
            return param_names, parameters

        for ensemble in ensembles:
            # if read_model_parameters returns empty lists, we shouldn't continue.
            # This is useful for empty ensembles where the user makes models.
            names, values = read_model_parameters(ensemble)
            if (len(names) != 0 and len(values) != 0):
                all_configs = self._permutation_strategy(
                    names, values, **kwargs)

                # run_settings can be ignored in this case as all models
                # will run with ensemble run_settings
                for i, conf in enumerate(all_configs):
                    model_name = "_".join((ensemble.name, str(i)))
                    m = NumModel(model_name, conf, None, run_settings={})
                    ensemble.add_model(m)

    def _create_experiment_dir(self, exp_path):
        """Creates the directory structure for the simulations"""

        if not path.isdir(exp_path):
            mkdir(exp_path)
        else:
            logger.info("Working in previously created experiment")


    def _create_orchestrator_dir(self, exp_path, orchestrator):
        """Generate the directory that will house the orchestrator output files"""

        if not orchestrator:
            return

        orc_path = path.join(exp_path, "orchestrator")
        orchestrator.set_path(orc_path)

        # remove orc files if present
        if path.isdir(orc_path):
            shutil.rmtree(orc_path)
        mkdir(orc_path)

    def _create_nodes(self, exp_path, nodes, node_files):
        """Generate the files and directories for SmartSimNodes"""

        if not nodes:
            return

        for node in nodes:
            node_path = path.join(exp_path, node.name)
            node.set_path(node_path)
            if path.isdir(node_path):
                shutil.rmtree(node_path)
            mkdir(node_path)

        for node_file in node_files:
            dst_path = path.join(node_path, path.basename(node_file))
            config_path = self.check_path(node_file)
            if path.isdir(config_path):
                dir_util.copy_tree(config_path, node_path)
            else:
                shutil.copyfile(config_path, dst_path)

    def _create_ensembles(self, exp_path, ensembles):
        """Generate the directories for the ensembles"""

        if not ensembles:
            return

        for ensemble in ensembles:
            ensemble_dir = path.join(exp_path, ensemble.name)
            if path.isdir(ensemble_dir):
                shutil.rmtree(ensemble_dir)
            mkdir(ensemble_dir)

    def _configure_models(self, ensembles, exp_path, model_files):
        """Based on the params argument to a Ensemble instance, read, configure
           and write model input files.
        """

        for ensemble in ensembles:

            # Make ensemble model directories
            for name, model in ensemble.models.items():
                dst = path.join(exp_path, ensemble.name, name)
                mkdir(dst)
                model.set_path(dst)

                if model_files:
                    for config in model_files:
                        dst_path = path.join(dst, path.basename(config))
                        config_path = self.check_path(config)
                        if path.isdir(config_path):
                            dir_util.copy_tree(config_path, dst)
                        else:
                            shutil.copyfile(config_path, dst_path)

                    # write in changes to configurations
                    self._writer.write(model)

            logger.info(f"Generated {len(ensemble)} models for ensemble: {ensemble.name}")

    def check_path(self, file_path):
        """Given a user provided path-like str, find the actual path to
           the directory or file and create a full path.
        """
        full_path = path.abspath(file_path)
        if path.isfile(full_path):
            return full_path
        elif path.isdir(full_path):
            return full_path
        else:
            raise SSConfigError(f"File or Directory {file_path} not found")