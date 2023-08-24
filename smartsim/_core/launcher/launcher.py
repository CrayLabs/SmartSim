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

import abc
import typing as t

from ...error import AllocationError, LauncherError, SSUnsupportedError
from .stepInfo import UnmanagedStepInfo, StepInfo
from .stepMapping import StepMapping
from .taskManager import TaskManager
from .step import Step
from ...settings import SettingsBase


class Launcher(abc.ABC):  # pragma: no cover
    """Abstract base class of all launchers

    This class provides the interface between the experiment
    controller and the launcher layer. Each launcher supported
    in SmartSim should implement the methods in this class to
    be fully compatible.
    """

    step_mapping: StepMapping
    task_manager: TaskManager

    @property
    @abc.abstractmethod
    def supported_rs(self) -> t.Dict[t.Type[SettingsBase], t.Type[Step]]:
        raise NotImplementedError

    @abc.abstractmethod
    def create_step(self, name: str, cwd: str, step_settings: SettingsBase) -> Step:
        raise NotImplementedError

    @abc.abstractmethod
    def get_step_update(
        self, step_names: t.List[str]
    ) -> t.List[t.Tuple[str, t.Union[StepInfo, None]]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_step_nodes(self, step_names: t.List[str]) -> t.List[t.List[str]]:
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, step: Step) -> t.Optional[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self, step_name: str) -> StepInfo:
        raise NotImplementedError


class WLMLauncher(Launcher):  # cov-wlm
    """The base class for any Launcher that utilizes workload
    manager specific commands. This base class is used to provide
    implemented methods that are alike across all WLM launchers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.task_manager = TaskManager()
        self.step_mapping = StepMapping()

    # every launcher utilizing this interface must have a map
    # of supported RunSettings types (see slurmLauncher.py for ex)
    def create_step(
        self, name: str, cwd: str, step_settings: SettingsBase
    ) -> Step:  # cov-wlm
        """Create a WLM job step

        :param name: name of the entity to be launched
        :type name: str
        :param cwd: path to launch dir
        :type cwd: str
        :param step_settings: batch or run settings for entity
        :type step_settings: BatchSettings | RunSettings
        :raises SSUnsupportedError: if batch or run settings type isnt supported
        :raises LauncherError: if step creation fails
        :return: step instance
        :rtype: Step
        """
        try:
            step_class = self.supported_rs[type(step_settings)]
        except KeyError:
            raise SSUnsupportedError(
                f"RunSettings type {type(step_settings)} not supported by this launcher"
            ) from None
        try:
            return step_class(name, cwd, step_settings)
        except AllocationError as e:
            raise LauncherError("Step creation failed") from e

    # these methods are implemented in WLM launchers and
    # don't need to be covered here.

    def get_step_nodes(
        self, step_names: t.List[str]
    ) -> t.List[t.List[str]]:  # pragma: no cover
        raise SSUnsupportedError("Node acquisition not supported for this launcher")

    def get_step_update(
        self, step_names: t.List[str]
    ) -> t.List[t.Tuple[str, t.Union[StepInfo, None]]]:  # cov-wlm
        """Get update for a list of job steps

        :param step_names: list of job steps to get updates for
        :type step_names: list[str]
        :return: list of name, job update tuples
        :rtype: list[(str, StepInfo)]
        """
        updates: t.List[t.Tuple[str, t.Union[StepInfo, None]]] = []

        # get updates of jobs managed by workload manager (PBS, Slurm, etc)
        # this is primarily batch jobs.
        s_names, step_ids = self.step_mapping.get_ids(step_names, managed=True)
        if len(step_ids) > 0:
            _step_ids = [str(sid) for sid in step_ids]
            s_statuses = self._get_managed_step_update(_step_ids)
            if s_statuses:
                _updates = list(zip(s_names, s_statuses))
                updates.extend(_updates)

        # get updates of unmanaged jobs (Aprun, mpirun, etc)
        # usually jobs started and monitored through the Popen interface
        t_names, task_ids = self.step_mapping.get_ids(step_names, managed=False)
        if len(task_ids) > 0:
            _task_ids = [str(tid) for tid in task_ids]
            t_statuses = self._get_unmanaged_step_update(_task_ids)
            _updates = list(zip(t_names, t_statuses))
            updates.extend(_updates)

        return updates

    def _get_unmanaged_step_update(
        self, task_ids: t.List[str]
    ) -> t.List[UnmanagedStepInfo]:  # cov-wlm
        """Get step updates for Popen managed jobs

        :param task_ids: task id to check
        :type task_ids: list[str]
        :return: list of step updates
        :rtype: list[StepInfo]
        """
        updates = []
        for task_id in task_ids:
            stat, return_code, out, err = self.task_manager.get_task_update(task_id)
            update = UnmanagedStepInfo(stat, return_code, out, err)
            updates.append(update)
        return updates

    # pylint: disable-next=no-self-use
    def _get_managed_step_update(
        self,
        step_ids: t.List[str], # pylint: disable=unused-argument
    ) -> t.List[StepInfo]:  # pragma: no cover
        return []
