import time
import numpy as np
import os.path as osp

class Step:
    def __init__(self, name, cwd):
        self.name = self._create_unique_name(name)
        self.entity_name = name
        self.cwd = cwd
        self.managed = False

    def get_launch_cmd(self):
        raise NotImplementedError

    def _create_unique_name(self, entity_name):
        step_name = entity_name + "-" + str(np.base_repr(time.time_ns(), 36))
        return step_name

    def get_output_files(self):
        """Return two paths to error and output files based on cwd"""
        output = self.get_step_file(ending=".out")
        error = self.get_step_file(ending=".err")
        return output, error

    def get_step_file(self, ending=".sh"):
        """Get the name for a file/script created by the step class

        Used for Batch scripts, mpmd scripts, etc"""
        return osp.join(self.cwd, self.entity_name + ending)
