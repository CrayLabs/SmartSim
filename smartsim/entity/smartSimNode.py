import os
from os.path import join
from .entity import SmartSimEntity


class SmartSimNode(SmartSimEntity):
    def __init__(self, name, path, run_settings=dict()):
        super().__init__(name, path, "node", run_settings)
