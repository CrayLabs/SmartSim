from .error import SmartSimError

class NumModel:
    """Hold configuration data for a numerical model. This
       class is passed around in the data-generation stage of
       the SS pipeline so that the configurations can be read
       easily.
    """

    def __init__(self, name, param_dict, path=None):
        self.name = name
        if type(param_dict) != dict:
            raise SmartSimError("Model must be initialized with parameter dictionary!  param_dict is: " + str(param_dict))
        self.param_dict = param_dict
        if path:
            self.set_path(path)

    def get_path(self):
        return self.path

    def set_path(self, path):
        self.path = path
        
    def get_param_value(self, param):
        return self.param_dict[param]
