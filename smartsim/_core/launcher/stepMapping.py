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

from ...log import get_logger

logger = get_logger(__name__)


class StepMap:
    def __init__(self,
                 step_id: t.Optional[str] = None,
                 task_id: t.Optional[str] = None,
                 managed: t.Optional[bool] = None) -> None:
        self.step_id = step_id
        self.task_id = task_id
        self.managed = managed


class StepMapping:
    def __init__(self) -> None:
        # step_name : wlm_id, pid, wlm_managed?
        self.mapping: t.Dict[str, StepMap] = {}

    def __getitem__(self, step_name: str) -> StepMap:
        return self.mapping[step_name]

    def __setitem__(self, step_name: str, step_map: StepMap) -> None:
        self.mapping[step_name] = step_map

    def add(
        self,
        step_name: str,
        step_id: t.Optional[str] = None,
        task_id: t.Optional[str] = None,
        managed: bool = True,
    ) -> None:
        try:
            n_task_id = str(task_id) if task_id else None
            self.mapping[step_name] = StepMap(step_id, n_task_id, managed)
        except Exception as e:
            msg = f"Could not add step {step_name} to mapping: {e}"
            logger.exception(msg)

    def get_task_id(self, step_id: str) -> t.Optional[str]:
        """Get the task id from the step id"""
        task_id = None
        for stepmap in self.mapping.values():
            if stepmap.step_id == step_id:
                task_id = stepmap.task_id
                break
        return task_id

    def get_ids(
        self, step_names: t.List[str], managed: bool = True
    ) -> t.Tuple[t.List[str], t.List[t.Union[str, None]]]:
        ids: t.List[t.Union[str, None]] = []
        names = []
        for name in step_names:
            if name in self.mapping:
                stepmap = self.mapping[name]
                # do we want task(unmanaged) or step(managed) id?
                if managed and stepmap.managed:
                    names.append(name)
                    ids.append(stepmap.step_id)
                elif not managed and not stepmap.managed:
                    names.append(name)
                    s_task_id = str(stepmap.task_id) if stepmap.task_id else None
                    ids.append(s_task_id)
        return names, ids
