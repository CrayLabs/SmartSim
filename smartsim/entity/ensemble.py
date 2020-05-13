
from os import path, mkdir
from ..error import SSModelExistsError
from .entity import SmartSimEntity

class Ensemble(SmartSimEntity):
    """Ensembles are groups of NumModels that can be used
       for quickly generating a number of models with different
       model parameter spaces.

       Models within the default Ensemble will use their own
       run_settings, whereas models not in the default ensemble
       will inheirt the run_settings of that ensemble.
    """

    def __init__(self, name, params, experiment, path, run_settings={}):
        """Initalize an Ensemble of NumModel instances.

        :param name: Name of the ensemble
        :type name: str
        :param params: model parameters for NumModel generation
        :type params: dict
        :param experiment: name of the experiment
        :type experiment: str
        :param path: path to output, error and conf files
        :type path: str
        :param run_settings: settings for the launcher, defaults to {}
        :type run_settings: dict, optional
        """
        super().__init__(name, path, "ensemble", run_settings)
        self.params = params
        self.experiment = experiment
        self.models = {}

    def add_model(self, model):
        """Add a model to this ensemble

        :param model: model instance
        :type model: NumModel
        :raises SSModelExistsError: if model already exists in this ensemble
        """
        if model.name in self.models:
            raise SSModelExistsError("Model name: " + model.name +
                                     " already exists in ensemble: " + self.name)
        else:
            self.models[model.name] = model

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
