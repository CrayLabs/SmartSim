import argparse
from os import path
from shutil import which
from smartsim import Experiment
from smartsim.error import MPOError



class MPO():

    model_initialized = False
    if not which("srun"):
        raise MPOError(
            "MPO currently only supports Slurm as the SmartSim backend")

    def __init__(self, tunable_params, data_dir="MPO"):
        """Initalize a MPO instance for running SmartSim with CrayAI

        :param tunable_params: Model parameters to tune with
                               defaults
        :type tunable_params: dict
        :param data_dir: Data directory name, defaults to "MPO"
        :type data_dir: str, optional
        """
        self.experiment = Experiment(data_dir)
        self.eval_params = None
        self.alloc = None

        args = self._parse_parameters(tunable_params)
        self._setup_evaluation(args)

    def init_model(self, run_settings, model_params={}):
        """Initialize a model for evaluation. This should only
           be called once in an evaluation script for CrayAI.

        :param run_settings: how the model should run on SmartSim backend
        :type run_settings: dict
        :param model_params: model parameters in addition to tunable
                             parameters, defaults to {}
        :type model_params: dict, optional
        :return: model instance
        :rtype: NumModel
        """
        # add allocation to model run_settings
        if self.alloc:
            run_settings["alloc"] = self.alloc
        model_params.update(self.eval_params)
        model_name = self._create_model_name()
        model = self.experiment.create_model(
            model_name,
            params=model_params,
            run_settings=run_settings
        )
        self.model_initialized = True
        return model

    def run(self, poll_interval=10):
        """Start the model initialized by MPO.init_model() and
           any other models, nodes, or orchestrators within the
           MPO experiment. This will usually just be the single
           model for evaluation. MPO.init_model() must be called
           prior to this function.

        :param poll_interval: how often to ping SmartSim backend
                              (workload manager) for status updates,
                              defaults to 10
        :type poll_interval: int, optional
        :raises MPOError: Raised if MPO.init_model() has not been called
        """
        if not self.model_initialized:
            raise MPOError(
                "Model has not been intialized for evaluation; call MPO.init_model()")
        self.experiment.generate()
        self.experiment.start()
        self.experiment.poll(interval=poll_interval)

    def get_model_file(self, file_name):
        """Retrieve a file path from the directory of the
           model created by MPO.init_model()

        :param file_name: name of the file to retrieve
        :type file_name: str
        :return: file path to the requested file
        :rtype: str
        """
        model_name = self._create_model_name()
        model = self.experiment.get_model(model_name, "default")
        return path.join(model.path, file_name)

    def get_eval_params(self):
        """Return the evaluation parameters provided by CrayAI
           for this model instance.

        :return: tunable model parameters
        :rtype: dict
        """
        return self.eval_params

    def _parse_parameters(self, tunable_params):
        """Parse the parameters provided by CrayAI

        :param tunable_params: tunable model parameters
        :type tunable_params: dict
        :raises TypeError: if tunable_params is not a dictionary
        :return: parsed model parameters
        :rtype: dict
        """
        if not isinstance(tunable_params, dict):
            raise TypeError(
                "Tunable parameters must be provided as a dictionary")

        argparser = argparse.ArgumentParser()
        argparser.add_argument("--alloc", type=str, default=None)
        for param, default in tunable_params.items():
            argparser.add_argument(f"--{param}",
                                   type=int,
                                   default=default)
        args = vars(argparser.parse_args())
        return args

    def _setup_evaluation(self, parsed_args):
        """Setup the allocation if necessary and set
           the tunable model parameters in the MPO
           class as "eval_params"

        :param parsed_args: args parsed from command line
        :type parsed_args: dict
        """
        alloc_id = parsed_args.pop("alloc")
        self.eval_params = parsed_args
        if alloc_id:
            self.experiment.add_allocation(alloc_id)
            self.alloc = alloc_id

    def _create_model_name(self):
        """Create a unique model name based on the
           tunable parameters

        :return: model name
        :rtype: str
        """
        name = ""
        for k, v in self.eval_params.items():
            name += f"{k}-{v}_"
        name = name.rstrip("_")
        return name
