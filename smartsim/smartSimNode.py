
import os

class SmartSimNode:

    def __init__(self, name, path=None, **kwargs):
        self.name = name
        self.settings = kwargs
        if path:
            self.path = path
        else:
            self.path = os.getcwd()

    def __str__(self):
        node_str = "\n   " + self.name + "\n"
        for param, value in self.settings.items():
            node_str += " ".join(("    ", str(param), "=" , str(value), "\n"))
        return node_str