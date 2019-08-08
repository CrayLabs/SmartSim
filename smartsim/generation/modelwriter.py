import re
import glob
import os

from os import path

from ..helpers import get_SSHOME
from ..error import SSConfigError, SmartSimError


class ModelWriter:

    def __init__(self):
        self.tag = '(;.+;)'

    def write(self, model, model_path):
        """Takes a model and writes the configuration to the tar_get_configs
           Base configurations are duplicated blindly to ensure all needed
           files are copied.
        """

        # configurations reset each time a new model is introduced
        for conf_path, _, files in os.walk(model_path):
            for fn in files:
                conf = path.join(conf_path, fn)
                if path.isfile(conf):
                    self._set_lines(conf)
                    self._replace_tags(model)
                    self._write_changes(conf)
                elif path.isdir(conf):
                    continue
                else:
                    raise SmartSimError("Data-Generation", "Could not find target configuration files")

    def _set_lines(self, conf_path):
        fp = open(conf_path, "r+")
        self.lines = fp.readlines()
        fp.close()

    def _write_changes(self, conf_path):
        """Write the target-specific changes"""
        fp = open(conf_path, "w+")
        for line in self.lines:
            fp.write(line)
        fp.close()

    def _replace_tags(self, model):
        """Adds the configurations specified in the regex syntax or the
           simulation.toml"""
        edited = []
        for line in self.lines:
            search = re.search(self.tag, line)
            if search:
                tagged_line = search.group(0)
                previous_value = self._get_prev_value(tagged_line)
                if self._is_target_spec(tagged_line, model.param_dict):
                    new_val = str(model.param_dict[previous_value])
                    new_line = re.sub(self.tag, new_val, line)
                    edited.append(new_line)

                # if a tag is found but is not in this model's configurations
                # put in placeholder value
                else:
                    edited.append(re.sub(self.tag, previous_value, line))
            else:
                edited.append(line)

        self.lines = edited


    def _is_target_spec(self, tagged_line, model_params):
        # NOTE: think about how this might work with tags
        split_tag = tagged_line.split(";")
        prev_val = split_tag[1]
        if prev_val in model_params.keys():
            return True
        return False

    def _get_prev_value(self, tagged_line):
        split_tag = tagged_line.split(";")
        return split_tag[1]

