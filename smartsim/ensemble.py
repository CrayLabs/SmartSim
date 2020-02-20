
from os import path, mkdir
from .error import SSModelExistsError
from .entity import SmartSimEntity

class Ensemble(SmartSimEntity):

    def __init__(self, name, params, experiment, path, run_settings={}):
        super().__init__(name, path, "ensemble", run_settings)
        self.params = params
        self.experiment = experiment
        self.models = {}

    def add_model(self, model):
        if model.name in self.models:
            raise SSModelExistsError("Model name: " + model.name +
                                     " already exists in ensemble: " + self.name)
        else:
            self.models[model.name] = model

    def __str__(self):
        ensemble_str = "\n   " + self.name + "\n"
        for model in self.models.values():
            ensemble_str += str(model)
        ensemble_str += "\n"
        return ensemble_str

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
