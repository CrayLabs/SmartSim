from os import path, mkdir
from ..error import EntityExistsError, SmartSimError
from .entity import SmartSimEntity
from .files import EntityFiles

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
        self._key_prefixing_enabled = True
        self.models = {}

    def add_model(self, model, overwrite=False):
        """Add a model to this ensemble

        :param model: model instance
        :type model: NumModel
        :param overwrite: overwrite model if it already exists
        :raises EntityExistsError: if model already exists in this ensemble
        """
        if model.name in self.models and not overwrite:
            raise EntityExistsError(
                f"Model {model.name} already exists in ensemble {self.name}")
        else:
            # Ensemble members need a key_prefix set to avoid namespace clashes
            self.models[model.name] = model
            if self.name != 'default':
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

    def enable_key_prefixing(self):
        """If called, all models within this ensemble will prefix their keys with its
        own model name.
        """
        for model in self.models:
            model.enable_key_prefixing()

    def disable_key_prefixing(self):
        """This function should not be called for SmartSim ensemble instances to avoid
        key namespace clashes
        """
        raise SmartSimError("Ensembles should never have key prefixing disabled")

    def query_key_prefixing(self):
        """Inquire as to whether each model within the ensemble will prefix its keys
        :returns: True if all models have key prefixing enabled, False otherwise
        :rtype: dict
        """
        return all([model.query_key_prefixing() for model in self.models])

    def __str__(self):
        ensemble_str = f"\nEnsemble: {self.name}"
        if len(self.models) < 1:
            return super().__str__()

        for model in self.models.values():
            ensemble_str += "\n" + str(model)
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
