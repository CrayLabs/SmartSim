
from ..ssModule import SSModule
from abc import ABC, abstractmethod


class Processor(SSModule, ABC):
    """Converts multiple files of simulation data into
       datasets convenient for machine learning experiments"""

    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self.datasets = []


###########################
### Processor Interface ###
###########################

    @abstractmethod
    def get_datasets(self):
        """Return a single dataset from each target"""
        
    @abstractmethod
    def _set_datasets(self):
        """Open the datasets for the processor to work with"""

###########################