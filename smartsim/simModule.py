import sys

from .error import SSConfigError, SmartSimError
from .coloredLogger import _get_logger

class SmartSimModule:
    """The base class of all the modules within SmartSim. The SmartSim
       modules as it stands today are:
         - Generator
         - Controller

       Each of the SmartSimModule children have access to the State
       information through this class.

       :param State state: State Instance
    """

    def __init__(self, state, name=__name__, log_level="DEV", **kwargs):
        self.state = state
        self._init_args = kwargs
        self.logger = _get_logger(name, log_level)

    def get_state(self):
        """Return the current state of the experiment

           :returns: A string corresponding to the current state

        """
        return self.state.current_state

    def get_targets(self):
        """Get a list of the targets created by the user.

           :returns: List of targets in the State instance
        """
        return self.state.targets

    def get_target(self, target):
        """Return a specific target from State

           :param str target: Name of the target to return

           :returns: Target instance
           :raises: SmartSimError
        """
        for t in self.state.targets:
            if t.name == target:
                return t
        raise SmartSimError(self.get_state(), "Target not found: " + target)

    def get_nodes(self):
        """Get a list of the nodes declared in State

            :returns: list of SmartSimNode instances
        """
        return self.state.nodes

    def get_model(self, model, target):
        """Get a specific model from a target.

           :param str model: name of the model to return
           :param str target: name of the target where the model is located

           :returns: NumModel instance
        """
        try:
            target = self.get_target(target)
            model = target.get_model[model]
            return model
        # if the target is not found
        except SmartSimError:
            raise
        except KeyError:
            raise SmartSimError(self.get_state(), "Model not found: " + model)

    def get_experiment_path(self):
        """Get the path to the experiment where all the targets and models are
           held.

           :returns: Path to experiment
        """
        return self.state._get_expr_path()


    def get_config(self, conf_param, none_ok=False):
        to_find = conf_param
        if isinstance(to_find, list):
            to_find = conf_param[-1]
            if to_find in self._init_args.keys():
                return self._init_args[to_find]
        # if not in init args search simulation.toml
        return self.state._get_toml_config(conf_param, none_ok=none_ok)


    def has_orcestrator(self):
        """Has the orchestrator been initialized by the user"""
        if self.state.orc:
            return True
        else:
            return False

    def set_state(self, new_state):
        self.state.current_state = new_state
