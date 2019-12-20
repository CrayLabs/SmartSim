import os
from os.path import join
from .entity import SmartSimEntity

class SmartSimNode(SmartSimEntity):

    def __init__(self, name, path=None, run_settings=dict()):
        super().__init__(name, path, run_settings)
        if not path:
            self.path = os.getcwd()
        else:
            self.path = path

    def __str__(self):
        node_str = "\n   " + self.name + "\n"
        for param, value in self.run_settings.items():
            node_str += " ".join(("    ", str(param), "=" , str(value), "\n"))
        return node_str