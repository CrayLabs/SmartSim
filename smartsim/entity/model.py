from ..error import SmartSimError
from .entity import SmartSimEntity

class NumModel(SmartSimEntity):
    """One instance of a model. This class holds the various information to configure
       and run the model. A model can be created through experiment.create_model()."""

    def __init__(self, name, params, path, run_settings):
        """
        NumModel initializer

        :param str name: name of the model instance
        :param dict params: parameters of the model to be written into model configuration
                            files
        :param str path: desired path to the model files/data created at runtime
        :param dict run_settings: launcher settings for workload manager or local call
                                   e.g. {"ppn": 1, "nodes": 10, "partition":"default_queue"}
        """
        super().__init__(name, path, "model", run_settings)
        if type(params) != dict:
            raise SmartSimError("Model must be initialized with parameter dictionary!  params are: " + str(params))
        self.params = params


    def get_param_value(self, param):
        return self.params[param]

    def __eq__(self, other):
        if self.params == other.params:
            return True
        return False

    def __str__(self):
        model_str = "     " + self.name + "\n"
        for param, value in self.params.items():
            model_str += "       " + str(param) + " = " + str(value)  + "\n"
        return model_str