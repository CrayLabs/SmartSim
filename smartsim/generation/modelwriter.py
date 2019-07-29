

import re

from helpers import get_SSHOME
from error import SSConfigError, SmartSimError
from os.path import isfile



class ModelWriter:

    def __init__(self, target_configs):
        self.confs = target_configs
        self.tag = '(;.+;{.+};.+;)'

    def write(self, model, model_path):
        """Takes a model and writes the configuration to the target_configs"""
        for conf in self.confs:
            conf_path = "/".join((model_path, conf))
            if isfile(conf_path):
                self._set_lines(conf_path)
                self._replace_tags(model)
                self._write_changes(conf_path)
            else:
                # TODO write a better message
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
                    new_val = model.param_dict[previous_value]
                    edited.append(re.sub(self.tag, new_val, line))

                # if a tag is found but is not in this model's configurations
                # put in placeholder value
                else:
                    edited.append(re.sub(self.tag, previous_value, line))
            else:
                edited.append(line)

        self.lines = edited


    def _is_target_spec(self, tagged_line, model_params):
        split_tag = tagged_line.split(";")
        prev_val = split_tag[1]
        if prev_val in model_params.keys():
            return True
        return False

    def _get_prev_value(self, tagged_line):
        split_tag = tagged_line.split(";")
        return split_tag[1]

