

class NumModel:
    """Hold configuration data for a numerical model. This
       class is passed around in the data-generation stage of
       the MPO pipeline so that the configurations can be read
       easily.
    """

    def __init__(self, param_dict, settings):
        self.param_dict = param_dict
        self.settings = settings
        self.name = "Model"
        for param in param_dict.values():
            self.name += "_" + str(param)
