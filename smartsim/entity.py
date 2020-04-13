import os
from os.path import join

from smartsim.error.errors import SSConfigError


class SmartSimEntity:
    def __init__(self, name, path, entity_type, run_settings):
        self.name = name
        if path:
            self.path = path
        else:
            self.path = os.getcwd()
        self.type = entity_type
        self.run_settings = run_settings
        self._init_run_settings()
        self.cmd = ""

    def _init_run_settings(self):
        defaults = {
            "nodes": 1,
            "ppn": 1,
            "partition": None
        }
        defaults["cwd"] = self.path
        defaults["out_file"] = join(self.path, self.name + ".out")
        defaults["err_file"] = join(self.path, self.name + ".err")
        defaults.update(self.run_settings)
        self.run_settings = defaults

    def update_run_settings(self, update_dict):
        """Update the run settings of an entity but keep the path the same"""
        old_path = self.path
        self.run_settings.update(update_dict)
        self.set_path(old_path)

    def set_path(self, new_path):
        self.path = new_path
        self.run_settings["cwd"] = self.path
        self.run_settings["out_file"] = join(self.path, self.name + ".out")
        self.run_settings["err_file"] = join(self.path, self.name + ".err")
