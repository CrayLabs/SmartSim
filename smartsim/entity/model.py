from ..error import SmartSimError
from .entity import SmartSimEntity

class NumModel(SmartSimEntity):
    """A NumModel holds the information necessary to launch a model
       onto whichever launcher is specified in the experiment.
       NumModel is a subclass of SmartSimEntity and has the same
       defaults as every other entity within SmartSim.

       NumModels also have a params argument which allows models
       to be a part of an ensemble. The params argument are model
       parameters that can be written into model configuration files
       through the generator.
    """
    def __init__(self, name, params, path, run_settings):
        """Initialize a model entity within Smartsim

        :param name: name of the model
        :type name: str
        :param params: model parameters for writing into configuration files.
        :type params: dict
        :param path: path to output, error, and configuration files
                     at runtime
        :type path: str
        :param run_settings: settings for the launcher specified in
                             the experiment
        :type run_settings: dict
        """
        super().__init__(name, path, "model", run_settings)
        self.params = params

    def get_param_value(self, param):
        """Get a value of a model parameter

        :param param: parameter name
        :type param: str
        :return: value of the model paramter
        :rtype: str
        """
        return self.params[param]

    def __eq__(self, other):
        if self.params == other.params:
            return True
        return False
