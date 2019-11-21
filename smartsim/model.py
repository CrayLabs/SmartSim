from .error import SmartSimError

class NumModel:
    """Hold configuration data for a numerical model. This
       class is passed around in the data-generation stage of
       the SS pipeline so that the configurations can be read
       easily.
    """

    def __init__(self, name, params, path=None):
        self.name = name
        if type(params) != dict:
            raise SmartSimError("Model must be initialized with parameter dictionary!  params are: " + str(params))
        self.params = params
        if path:
            self.path = path

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