
from os import path, mkdir
from .error import SSModelExistsError

class Target:

    def __init__(self, name, params, experiment_name, target_dir_path):
        self.name = name
        self.params = params
        self.path = target_dir_path
        self.experiment = experiment_name
        self._models = {}


    def get_target_params(self):
        return self.params

    def get_target_dir(self):
        return self.path

    def get_models(self):
        return self._models

    def get_model(self, model_name):
        return self._models[model_name]

    def add_model(self, model):
        if model.name in self._models:
            raise SSModelExistsError("Adding model to target",
                                "Model name: " + model.name + " already exists in target: " + self.__repr__())
        else:
            self._models[model.name] = model

    def __str__(self):
        return self.name
