import sys

from .error import SSConfigError, SmartSimError

class SmartSimModule:
    """The base class of all the modules within SmartSim. The SmartSim
       modules as it stands today are:
         - Generator
         - Controller

       Each of the SmartSimModule children have access to the State
       information through this class.

       :param State state: State Instance
    """

    def __init__(self, state, **kwargs):
        self.state = state
        self._init_args = kwargs

    def get_state(self):
        """Return the current state of the experiment

           :returns: A string corresponding to the current state

        """
        return self.state.current_state

    def set_state(self, new_state):
        self.state.current_state = new_state

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
        return self.state.get_target(target)

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
        return self.state.get_model(model, target)

    def get_experiment_path(self):
        """Get the path to the experiment where all the targets and models are
           held.

           :returns: Path to experiment
        """
        return self.state.get_expr_path()


    def get_config(self, param, aux=None, none_ok=False):
        """Search for a configuration parameter in the initialization
           of a SmartSimModule. Also search through an auxiliry dictionary
           in some cases.

           :param str param: parameter to search for
           :param dict aux: auxiliry dictionary to search through (default=None)
           :param bool none_ok: ok to return none if param is not present (default=False)
           :raises KeyError:
           :returns: param if present
        """
        if aux and param in aux.keys():
            return aux[param]
        elif param in self._init_args.keys():
            return self._init_args[param]
        else:
            if none_ok:
                return None
            else:
                raise KeyError


    def has_orchestrator(self):
        """Has the orchestrator been initialized by the user"""
        if self.state.orc:
            return True
        else:
            return False
