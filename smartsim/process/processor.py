
import sys

from os import path
from abc import ABC, abstractmethod

from ..simModule import SmartSimModule
from ..helpers import get_SSHOME
from ..error import SmartSimError, SSConfigError


class Processor(SmartSimModule):
    """A convenient base case for constructing processors that can handle
       the data coming out of a simulation model. 
       
       The Processor class aims to make it easy for users to construct
       multiple processing stages for any type of problem/model.
    """

    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self.set_state("Data Processing")


###########################
### Processor Interface ###
###########################


    @abstractmethod
    def process(self):
        raise NotImplementedError

    def get_target_data(self, target, filename=None):
        """Look for generated datasets within a single target and return if present
           If not present, look through initialization arguments
        """
        data_paths = []
        target = self.get_target(target)
        data_file = self.get_config(["process","filename"])
        if filename:
            data_file = filename
        
        for name, model in target.get_models().items():
            data_path = path.join(model.get_path(), data_file)
            data_paths.append((name, data_path))
        if len(data_paths) < 1:
            raise SmartSimError(self.get_state(), "No data found for target: " + target.name)

        return data_paths

    def get_data(self, filename=None):
        """Look for generated datasets within targets and return if present
           If not present, look through initialization arguments
        """
        data_paths = []
        targets = self.get_targets()
        if len(targets) > 0:
            data_file = self.get_config(["process","filename"])
            if filename:
                data_file = filename
            
            for target in targets:
                for name, model in target.get_models().items():
                    data_path = path.join(model.get_path(), data_file)
                    data_paths.append((name, data_path))

            return data_paths

        else:
            raise SmartSimError(self.get_state(), "No target models found!")



###########################