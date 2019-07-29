

class NumModel:
    """Hold configuration data for a numerical model. This
       class is passed around in the data-generation stage of
       the SS pipeline so that the configurations can be read
       easily.
    """

    def __init__(self, name, param_dict):
        self.name = name
        self.param_dict = param_dict
        for param in param_dict.values():
            self.name += "_" + str(param)
