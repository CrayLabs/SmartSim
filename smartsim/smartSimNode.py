
import os

class SmartSimNode:

    def __init__(self, name, path=None, **kwargs):
        self.name = name
        self.settings = kwargs
        if path:
            self.path = path
        else:
            self.path = os.getcwd()

    def get_settings(self):
        return self.settings
