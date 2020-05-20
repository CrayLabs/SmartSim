
from os import path, mkdir
from ..error import SSModelExistsError
from .entity import SmartSimEntity

class Ensemble(SmartSimEntity):
    """Ensembles are groups of NumModels that can be used
       for quickly generating a number of models with different
       model parameter spaces.

       Models within the default Ensemble will use their own
       run_settings, whereas models not in the default ensemble
       will inherit the run_settings of that ensemble.
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
            # Ensemble members need a key_prefix set to avoid namespace clashes
            self.models[model.name] = model
            if self.name != 'default':
                model.key_prefix = f'{self.name}_{model.name}'
                model.enable_key_prefixing()

    def register_incoming_entity(self, incoming_entity, receiving_client_type):
        """Registers the named data sources that this entity has access to by storing
           the key_prefix associated with that entity

           Only python clients can have multiple incoming connections

           :param incoming_entity: The named SmartSim entity that the ensemble will
                                   receive data from
           :param type: SmartSimEntity
           :param receiving_client_type: The language of the SmartSim client used by
                                         this ensemble object. Can be cpp, fortran,
                                         python
           :param type: str
        """
        for model in self.models.values():
            model.register_incoming_entity(incoming_entity, receiving_client_type)

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
