import os
from os.path import join

from smartsim.error.errors import SSConfigError


class SmartSimEntity:
    def __init__(self, name, path, run_settings):
        self.name = name
        if path:
            self.path = path
        else:
            self.path = os.getcwd()
        self.run_settings = run_settings
        self._init_run_settings()
        self.cmd = ""

    def _init_run_settings(self):
        defaults = {
            "nodes": 1,
            "ppn": 1,
            "duration": "1:00:00",
            "partition": None
        }
        defaults["cwd"] = self.path
        defaults["out_file"] = join(self.path, self.name + ".out")
        defaults["err_file"] = join(self.path, self.name + ".err")
        defaults.update(self.run_settings)
        self.run_settings = defaults

    def get_run_settings(self):
        return self.run_settings

    def get_cmd(self):
        if not self.cmd:
            raise SSConfigError("Entity has not been configured to run")
        else:
            return self.cmd

    def update_run_settings(self, update_dict):
        self.run_settings.update(update_dict)

    def set_cmd(self, cmd):
        self.cmd = cmd

    def set_path(self, new_path):
        self.path = new_path
        self.run_settings["cwd"] = self.path
        self.run_settings["out_file"] = join(self.path, self.name + ".out")
        self.run_settings["err_file"] = join(self.path, self.name + ".err")
