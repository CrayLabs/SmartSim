# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

import os
import shutil

from .step import Step

class LocalStep(Step):
    def __init__(self, name, cwd, run_settings):
        super().__init__(name, cwd)
        self.run_settings = run_settings
        self.env = self._set_env()

    def get_launch_cmd(self):
        cmd = []

        # Add run command and args if user specified
        # default is no run command for local job steps
        if self.run_settings.run_command:
            cmd.append(self.run_settings.run_command)
            run_args = self.run_settings.format_run_args()
            cmd.extend(run_args)

        if self.run_settings.colocated_db_settings:
            # Replace the command with the entrypoint wrapper script
            bash = shutil.which("bash")

            launch_script_path = self.get_colocated_launch_script()
            cmd.extend([bash, launch_script_path])

        # build executable
        cmd.extend(self.run_settings.exe)
        if self.run_settings.exe_args:
            cmd.extend(self.run_settings.exe_args)

        return cmd

    def _set_env(self):
        env = os.environ.copy()
        if self.run_settings.env_vars:
            for k, v in self.run_settings.env_vars.items():
                env[k] = v
        return env
