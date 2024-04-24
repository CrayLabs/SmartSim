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

import pathlib
import typing as t

from ..._core.launcher.step import Step
from ...entity import EntityList, Model
from ...error import SmartSimError
from ..launcher.launcher import Launcher

if t.TYPE_CHECKING:
    from ..utils.serialize import TStepLaunchMetaData


class _AnonymousBatchJob(EntityList[Model]):
    @staticmethod
    def _validate(model: Model) -> None:
        if model.batch_settings is None:
            msg = "Unable to create _AnonymousBatchJob without batch_settings"
            raise SmartSimError(msg)

    def __init__(self, model: Model) -> None:
        self._validate(model)
        super().__init__(model.name, model.path)
        self.entities = [model]
        self.batch_settings = model.batch_settings

    def _initialize_entities(self, **kwargs: t.Any) -> None: ...


def _look_up_launched_data(
    launcher: Launcher,
) -> t.Callable[[t.Tuple[str, Step]], "TStepLaunchMetaData"]:
    def _unpack_launched_data(data: t.Tuple[str, Step]) -> "TStepLaunchMetaData":
        # NOTE: we cannot assume that the name of the launched step
        # ``launched_step_name`` is equal to the name of the step referring to
        # the entity ``step.name`` as is the case when an entity list is
        # launched as a batch job
        launched_step_name, step = data
        launched_step_map = launcher.step_mapping[launched_step_name]
        out_file, err_file = step.get_output_files()
        return (
            launched_step_map.step_id,
            launched_step_map.task_id,
            launched_step_map.managed,
            out_file,
            err_file,
            pathlib.Path(step.meta.get("status_dir", step.cwd)),
        )

    return _unpack_launched_data
