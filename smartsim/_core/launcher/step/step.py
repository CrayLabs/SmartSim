# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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

import functools
import os.path as osp
import pathlib
import sys
import time
import typing as t
from os import makedirs

from smartsim._core.config import CONFIG
from smartsim._core.types import StepName
from smartsim.error.errors import SmartSimError, UnproxyableStepError

from ....log import get_logger
from ....settings.base import RunSettings, SettingsBase
from ...utils.helpers import encode_cmd, get_base_36_repr
from ..colocated import write_colocated_launch_script

logger = get_logger(__name__)

if t.TYPE_CHECKING:
    from smartsim.entity import types as _entity_types


class Step:
    def __init__(
        self, name: "_entity_types.EntityName", cwd: str, step_settings: SettingsBase
    ) -> None:
        self.name: t.Final = self._create_unique_name(name)
        self.entity_name: t.Final = name
        self.cwd: t.Final = cwd
        self.step_settings = step_settings
        self.meta: t.Final[t.Dict[str, str]] = {}

    @property
    def managed(cls) -> bool:
        # XXX: keeping this as an instance attr for now, this is really more
        #      a class property than an instance propery. Even better would
        #      be to have different classes for managed and unmanaged steps
        return False

    @property
    def env(self) -> t.Optional[t.Dict[str, str]]:
        """Overridable, read only property for step to specify its environment"""
        return None

    def get_launch_cmd(self) -> t.List[str]:
        raise NotImplementedError

    @staticmethod
    def _create_unique_name(entity_name: "_entity_types.EntityName") -> StepName:
        return StepName(f"{entity_name}-{get_base_36_repr(time.time_ns())}")

    @staticmethod
    def _ensure_output_directory_exists(output_dir: str) -> None:
        """Create the directory for the step output if it doesn't exist already"""
        if not osp.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    def get_output_files(self) -> t.Tuple[str, str]:
        """Return two paths to error and output files based on metadata directory"""
        try:
            output_dir = self.meta["status_dir"]
        except KeyError as exc:
            raise KeyError("Status directory for this step has not been set.") from exc
        self._ensure_output_directory_exists(output_dir)
        output = osp.join(output_dir, f"{self.entity_name}.out")
        error = osp.join(output_dir, f"{self.entity_name}.err")
        return output, error

    def get_step_file(
        self, ending: str = ".sh", script_name: t.Optional[str] = None
    ) -> str:
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
        :type step: Step
        """
        raise SmartSimError("add_to_batch not implemented for this step type")


_StepT = t.TypeVar("_StepT", bound=Step)


def proxyable_launch_cmd(
    fn: t.Callable[[_StepT], t.List[str]], /
) -> t.Callable[[_StepT], t.List[str]]:
    @functools.wraps(fn)
    def _get_launch_cmd(self: _StepT) -> t.List[str]:
        """
        Generate a launch command that executes the `JobStep` with the
        indirect launching entrypoint instead of directly. The original
        command is passed to the proxy as a base64 encoded string.

        Steps implementing `get_launch_cmd` and decorated with
        `proxyable_launch_cmd` will generate status updates that can be consumed
        by the telemetry monitor and dashboard"""
        original_cmd_list = fn(self)

        if not CONFIG.telemetry_enabled:
            return original_cmd_list

        if self.managed:
            raise UnproxyableStepError(
                f"Attempting to proxy managed step of type {type(self)}"
                "through the unmanaged step proxy entry point"
            )

        proxy_module = "smartsim._core.entrypoints.indirect"
        entity_type = self.meta["entity_type"]
        status_dir = self.meta["status_dir"]

        # encode the original cmd to avoid potential collisions and escaping
        # errors when passing it using CLI arguments to the indirect entrypoint
        encoded_cmd = encode_cmd(original_cmd_list)

        # return a new command that executes the proxy and passes
        # the original command as an argument
        return [
            sys.executable,
            "-m",
            proxy_module,
            "+name",
            self.name,
            "+command",
            encoded_cmd,
            "+entity_type",
            entity_type,
            "+telemetry_dir",
            status_dir,
            "+working_dir",
            self.cwd,
        ]

    return _get_launch_cmd
