# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time
import os.path as osp

from ..colocated import write_colocated_launch_script
from ...utils.helpers import get_base_36_repr

class Step:
    def __init__(self, name, cwd):
        self.name = self._create_unique_name(name)
        self.entity_name = name
        self.cwd = cwd
        self.managed = False

    def get_launch_cmd(self):
        raise NotImplementedError

    def _create_unique_name(self, entity_name):
        step_name = entity_name + "-" + get_base_36_repr(time.time_ns())
        return step_name

    def get_output_files(self):
        """Return two paths to error and output files based on cwd"""
        output = self.get_step_file(ending=".out")
        error = self.get_step_file(ending=".err")
        return output, error

    def get_step_file(self, ending=".sh", script_name=None):
        """Get the name for a file/script created by the step class

        Used for Batch scripts, mpmd scripts, etc"""
        if script_name:
            script_name = script_name if "." in script_name else script_name + ending
            return osp.join(self.cwd, script_name)
        return osp.join(self.cwd, self.entity_name + ending)

    def get_colocated_launch_script(self):
        # prep step for colocated launch if specifed in run settings
        script_path = self.get_step_file(script_name=".colocated_launcher.sh")
        db_log_file = self.get_step_file(ending="-db.log")

        # write the colocated wrapper shell script to the directory for this
        # entity currently being prepped to launch
        write_colocated_launch_script(script_path,
                                      db_log_file,
                                      self.run_settings.colocated_db_settings)
        return script_path
