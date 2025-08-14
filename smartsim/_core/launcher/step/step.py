# BSD 2-Clause License
#
# Copyright (c) 2021-2025, Hewlett Packard Enterprise
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

from __future__ import annotations

import copy
import os.path as osp
import pathlib
import time
from os import makedirs

from smartsim.error.errors import SmartSimError

from ....log import get_logger
from ....settings.base import RunSettings, SettingsBase
from ...utils.helpers import get_base_36_repr
from ..colocated import write_colocated_launch_script

logger = get_logger(__name__)


class Step:
    def __init__(self, name: str, cwd: str, step_settings: SettingsBase) -> None:
        self.name = self._create_unique_name(name)
        self.entity_name = name
        self.cwd = cwd
        self.managed = False
        self.step_settings = copy.deepcopy(step_settings)
        self.meta: dict[str, str] = {}

    @property
    def env(self) -> dict[str, str] | None:
        """Overridable, read only property for step to specify its environment"""
        return None

    def get_launch_cmd(self) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def _create_unique_name(entity_name: str) -> str:
        step_name = entity_name + "-" + get_base_36_repr(time.time_ns())
        return step_name

    @staticmethod
    def _ensure_output_directory_exists(output_dir: str) -> None:
        """Create the directory for the step output if it doesn't exist already"""
        if not osp.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    def get_output_files(self) -> tuple[str, str]:
        """Return two paths to error and output files based on metadata directory"""
        try:
            output_dir = self.meta["metadata_dir"]
        except KeyError as exc:
            raise KeyError("Status directory for this step has not been set.") from exc
        self._ensure_output_directory_exists(output_dir)
        output = osp.join(output_dir, f"{self.entity_name}.out")
        error = osp.join(output_dir, f"{self.entity_name}.err")
        return output, error

    def get_step_file(self, ending: str = ".sh", script_name: str | None = None) -> str:
        """Get the name for a file/script created by the step class

        Used for Batch scripts, mpmd scripts, etc.
        """
        if script_name:
            script_name = script_name if "." in script_name else script_name + ending
            return osp.join(self.cwd, script_name)
        return osp.join(self.cwd, self.entity_name + ending)

    def get_colocated_launch_script(self) -> str:
        # prep step for colocated launch if specifed in run settings
        script_path = self.get_step_file(
            script_name=osp.join(
                ".smartsim", f"colocated_launcher_{self.entity_name}.sh"
            )
        )
        makedirs(osp.dirname(script_path), exist_ok=True)

        db_settings = {}
        if isinstance(self.step_settings, RunSettings):
            db_settings = self.step_settings.colocated_db_settings or {}

        # db log file causes write contention and kills performance so by
        # default we turn off logging unless user specified debug=True
        if db_settings.get("debug", False):
            db_log_file = self.get_step_file(ending="-db.log")
        else:
            db_log_file = "/dev/null"

        # write the colocated wrapper shell script to the directory for this
        # entity currently being prepped to launch
        write_colocated_launch_script(script_path, db_log_file, db_settings)
        return script_path

    # pylint: disable=no-self-use
    def add_to_batch(self, step: Step) -> None:
        """Add a job step to this batch

        :param step: a job step instance e.g. SrunStep
        """
        raise SmartSimError("add_to_batch not implemented for this step type")
