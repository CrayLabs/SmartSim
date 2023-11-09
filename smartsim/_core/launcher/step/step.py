# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import os.path as osp
import sys
import time
import typing as t
from os import makedirs

from smartsim.error.errors import SmartSimError

from ....log import get_logger
from ...utils.helpers import get_base_36_repr, encode_cmd
from ..colocated import write_colocated_launch_script
from ....settings.base import RunSettings, SettingsBase

logger = get_logger(__name__)


class Step:
    def __init__(self, name: str, cwd: str, step_settings: SettingsBase) -> None:
        self.name = self._create_unique_name(name)
        self.entity_name = name
        self.cwd = cwd
        self.managed = False
        self.step_settings = step_settings
        self.meta: t.Dict[str, str] = {}

    @property
    def env(self) -> t.Optional[t.Dict[str, str]]:
        """Overridable, read only property for step to specify its environment"""
        return None

    def get_launch_cmd(self) -> t.List[str]:
        raise NotImplementedError

    @staticmethod
    def _create_unique_name(entity_name: str) -> str:
        step_name = entity_name + "-" + get_base_36_repr(time.time_ns())
        return step_name

    def get_output_files(self) -> t.Tuple[str, str]:
        """Return two paths to error and output files based on cwd"""
        output = self.get_step_file(ending=".out")
        error = self.get_step_file(ending=".err")
        return output, error

    def get_step_file(
        self, ending: str = ".sh", script_name: t.Optional[str] = None
    ) -> str:
        """Get the name for a file/script created by the step class

        Used for Batch scripts, mpmd scripts, etc"""
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

        db_settings: t.Dict[str, str] = {}
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


# TODO: We should really consider adding ``typing_extensions`` as a dependency
#       To avoid needing to do this hacky solution for type hinting ``Self``
_TSelfUnmanagedProxyStep = t.TypeVar(
    "_TSelfUnmanagedProxyStep", bound="UnmanagedProxyStep"
)


class UnmanagedProxyStep(Step):
    def __init__(
        self,
        name: str,
        command: t.Sequence[str],
        cwd: str,
        settings: SettingsBase,
        proxy_cmd_output_files: t.Tuple[str, str],
        proxy_env: t.Optional[t.Dict[str, str]],
    ) -> None:
        super().__init__(name, cwd, settings)
        self._proxy_cmd_output_files = proxy_cmd_output_files
        self._env = proxy_env.copy() if proxy_env is not None else None
        self._wrapped_command = tuple(command)

    @classmethod
    def from_step(
        cls: t.Type[_TSelfUnmanagedProxyStep], step: Step
    ) -> _TSelfUnmanagedProxyStep:
        proxy_step = cls(
            step.entity_name,
            step.get_launch_cmd(),
            step.cwd,
            step.step_settings,
            step.get_output_files(),
            step.env,
        )
        proxy_step.meta = step.meta.copy()
        if step.managed:
            logger.debug(
                f"Managed step of type {type(step)} is has been cast to a {cls}"
            )
        return proxy_step

    @property
    def managed(self) -> bool:
        """Unmanaged steps will always be not managed"""
        return False

    @managed.setter
    def managed(self, val: bool) -> None:
        """Annoying no-op setter bc ``managed`` is not an abstract property in
        the ``Step`` interface
        """
        # TODO: This method should not exist and this attr
        # should be read only (I would argue same for base class)

    @property
    def env(self) -> t.Optional[t.Dict[str, str]]:
        return self._env

    @property
    def wrapped_command(self) -> t.Tuple[str, ...]:
        return self._wrapped_command

    def get_output_files(self) -> t.Tuple[str, str]:
        """Return two paths to error and output files based on cwd"""
        output = self.get_step_file(ending=".indirect.out")
        error = self.get_step_file(ending=".indirect.err")
        return output, error

    def get_launch_cmd(self) -> t.List[str]:
        """Executes a step indirectly through a proxy process. This ensures unmanaged tasks
        continue telemetry logging after a driver process exits or fails.

        :param step: the step to produce a proxied command for
        :type step: Step
        :return: CLI arguments to execute the step via the proxy step executor
        :rtype: t.List[str]
        """

        proxy_module = "smartsim._core.entrypoints.indirect"
        etype = self.meta["entity_type"]
        status_dir = self.meta["status_dir"]
        cmd_list = self.wrapped_command
        encoded_cmd = encode_cmd(list(cmd_list))
        out, err = self._proxy_cmd_output_files

        # note: this is NOT safe. should either 1) sign cmd and verify OR 2) serialize step and let
        # the indirect entrypoint rebuild the cmd... for now, test away...
        return [
            sys.executable,
            "-m",
            proxy_module,
            "+command",
            encoded_cmd,
            "+entity_type",
            etype,
            "+telemetry_dir",
            status_dir,
            "+working_dir",
            self.cwd,
            "+output_file",
            out,
            "+error_file",
            err,
        ]
