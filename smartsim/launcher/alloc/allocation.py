
from abc import ABC, abstractmethod

class Allocation(ABC):

    def __init__(self, nodes=1, ppn=1, duration="1:00:00", **kwargs):
        super().__init__()
        self.steps = 0
        self.nodes = nodes
        self.ppn = ppn
        self.duration = duration
        self.add_opts = kwargs


    @abstractmethod
    def __str__(self):
        """Return user-readable string form of allocation"""
        pass

    @abstractmethod
    def __repr__(self):
        """return parseable representation of allocation"""
        pass

    @abstractmethod
    def get_alloc_cmd(self):
        """Return the command to request an allocation"""
        pass



