
from os import path, mkdir
from .error import SSModelExistsError

class Target:

    def __init__(self, name, params, experiment, path, run_settings=None):
        self.name = name
        self.params = params
        self.path = path
        self.experiment = experiment
        self.models = {}
        if run_settings:
            self.run_settings = run_settings
        else:
            self.run_settings = dict()

    def add_model(self, model):
        if model.name in self.models:
            raise SSModelExistsError("Model name: " + model.name +
                                     " already exists in target: " + self.name)
        else:
            self.models[model.name] = model

    def __str__(self):
        target_str = "\n   " + self.name + "\n"
        for model in self.models.values():
            target_str += str(model)
        target_str += "\n"
        return target_str

    def __eq__(self, other):
        for model_1, model_2 in zip(self.models.values(),
                                    other.models.values()):
            if model_1 != model_2:
                return False
        return True

    def __getitem__(self, model_name):
        return self.models[model_name]

    def __len__(self):
        return len(self.models.values())
