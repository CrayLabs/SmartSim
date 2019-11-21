import re
import glob
import os

from os import path

from ..helpers import get_SSHOME
from ..error import SSConfigError, SmartSimError
from ..utils import get_logger
logger = get_logger(__name__)


class ModelWriter:

    def __init__(self):
        self.tag = ";"
        self.regex = "(;.+;)"

    def write(self, model):
        """Takes a model and writes the configuration to the target configs
           Base configurations are duplicated blindly to ensure all needed
           files are copied.
        """

        # configurations reset each time a new model is introduced
        for conf_path, _, files in os.walk(model.path):
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

    def _set_tag(self, tag, regex=None):
        if regex:
            self.regex = regex
        else:
            self.tag = tag
            self.regex = "".join(('(',tag,".+", tag, ')'))


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
        unused_tags = {}
        for i, line in enumerate(self.lines):
            search = re.search(self.regex, line)
            if search:
                tagged_line = search.group(0)
                previous_value = self._get_prev_value(tagged_line)
                if self._is_target_spec(tagged_line, model.params):
                    new_val = str(model.params[previous_value])
                    new_line = re.sub(self.regex, new_val, line)
                    edited.append(new_line)

                # if a tag is found but is not in this model's configurations
                # put in placeholder value
                else:
                    tag = tagged_line.split(self.tag)[1]
                    if tag not in unused_tags:
                        unused_tags[tag] = []
                    unused_tags[tag].append(i+1)
                    edited.append(re.sub(self.regex, previous_value, line))
            else:
                edited.append(line)
        for tag in unused_tags.keys():
            logger.warning("TAG: " + tag + " unused on line(s): " + str(unused_tags[tag]))
        self.lines = edited


    def _is_target_spec(self, tagged_line, model_params):
        split_tag = tagged_line.split(self.tag)
        prev_val = split_tag[1]
        if prev_val in model_params.keys():
            return True
        return False

    def _get_prev_value(self, tagged_line):
        split_tag = tagged_line.split(self.tag)
        return split_tag[1]

