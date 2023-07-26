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

import typing as t

from ..launcher import Launcher
from ....log import get_logger
from ....settings import RunSettings, SettingsBase
from ..step import LocalStep
from ..step import Step
from ..stepInfo import UnmanagedStepInfo, StepInfo
from ..stepMapping import StepMapping
from ..taskManager import TaskManager

logger = get_logger(__name__)


class LocalLauncher(Launcher):
    """Launcher used for spawning proceses on a localhost machine."""

    @property
    def supported_rs(self) -> t.Dict[t.Type[SettingsBase], t.Type[Step]]:
       return {
            RunSettings: LocalStep,
        }    
    
    def __init__(self) -> None:
        self.task_manager = TaskManager()
        self.step_mapping = StepMapping()

    def create_step(self, name: str, cwd: str, step_settings: SettingsBase) -> Step:
        """Create a job step to launch an entity locally

        :return: Step object
        """
        if not isinstance(step_settings, RunSettings):
            raise TypeError(
                f"Local Launcher only supports entities with RunSettings, not {type(step_settings)}"
            )
        step = LocalStep(name, cwd, step_settings)
        return step

    def get_step_update(self, step_names: t.List[str]) -> t.List[t.Tuple[str, t.Optional[StepInfo]]]:
        """Get status updates of each job step name provided

        :param step_names: list of step_names
        :type step_names: list[str]
        :return: list of tuples for update
        :rtype: list[(str, UnmanagedStepInfo)]
        """
        # step ids are process ids of the tasks
        # as there is no WLM intermediary
        updates: t.List[t.Tuple[str, t.Optional[StepInfo]]] = []
        s_names, s_ids = self.step_mapping.get_ids(step_names, managed=False)
        for step_name, step_id in zip(s_names, s_ids):
            status, rc, out, err = self.task_manager.get_task_update(str(step_id))
            step_info = UnmanagedStepInfo(status, rc, out, err)
            update = (step_name, step_info)
            updates.append(update)
        return updates

    def get_step_nodes(self, step_names: t.List[str]) -> t.List[t.List[str]]:
        """Return the address of nodes assigned to the step

        TODO: Use socket to find the actual Lo address?
        :return: a list containing the local host address
        """
        return [["127.0.0.1"] * len(step_names)]

    def run(self, step: Step) -> str:
        """Run a local step created by this launcher. Utilize the shell
           library to execute the command with a Popen. Output and error
           files will be written to the entity path.

        :param step: LocalStep instance to run
        :type step: LocalStep
        :return: task_id of the newly created step
        :rtype: str
        """
        if not self.task_manager.actively_monitoring:
            self.task_manager.start()

        out, err = step.get_output_files()
        output = open(out, "w+")
        error = open(err, "w+")
        cmd = step.get_launch_cmd()

        # LocalStep.run_command omits env, include it here
        passed_env = step.env if isinstance(step, LocalStep) else None

        task_id = self.task_manager.start_task(
            cmd, step.cwd, env=passed_env, out=output.fileno(), err=error.fileno()
        )
        self.step_mapping.add(step.name, task_id=task_id, managed=False)
        return task_id

    def stop(self, step_name: str) -> UnmanagedStepInfo:
        """Stop a job step

        :param step_name: name of the step to be stopped
        :type step_name: str
        :return: a UnmanagedStepInfo instance
        :rtype: UnmanagedStepInfo
        """
        # step_id is task_id for local. Naming for consistency
        step_id = self.step_mapping[step_name].task_id
        
        self.task_manager.remove_task(str(step_id))
        _, rc, out, err = self.task_manager.get_task_update(str(step_id))
        step_info = UnmanagedStepInfo("Cancelled", rc, out, err)
        return step_info

    def __str__(self) -> str:
        return "Local"
