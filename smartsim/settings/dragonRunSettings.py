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

import typing as t

from typing_extensions import override

from ..log import get_logger
from .base import RunSettings

logger = get_logger(__name__)


class DragonRunSettings(RunSettings):
    def __init__(
        self,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.List[str]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialize run parameters for a Dragon process

        ``DragonRunSettings`` should only be used on systems where Dragon
        is available and installed in the current environment.

        If an allocation is specified, the instance receiving these run
        parameters will launch on that allocation.

        :param exe: executable to run
        :param exe_args: executable arguments, defaults to None
        :param env_vars: environment variables for job, defaults to None
        :param alloc: allocation ID if running on existing alloc, defaults to None
        """
        super().__init__(
            exe,
            exe_args,
            run_command="",
            env_vars=env_vars,
            **kwargs,
        )

    @override
    def set_nodes(self, nodes: int) -> None:
        """Set the number of nodes

        :param nodes: number of nodes to run with
        """
        self.run_args["nodes"] = nodes

    @override
    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks for this job

        :param tasks_per_node: number of tasks per node
        """
        self.run_args["tasks-per-node"] = tasks_per_node

    @override
    def set_node_feature(self, feature_list: t.Union[str, t.List[str]]) -> None:
        """Specify the node feature for this job

        :param feature_list: a collection of strings representing the required
         node features. Currently supported node features are: "gpu"
        """
        if isinstance(feature_list, str):
            feature_list = feature_list.strip().split()
        elif not all(isinstance(feature, str) for feature in feature_list):
            raise TypeError("feature_list must be string or list of strings")

        self.run_args["node-feature"] = ",".join(feature_list)

    def set_cpu_affinity(self, devices: t.List[int]) -> None:
        """Set the CPU affinity for this job

        :param devices: list of CPU indices to execute on
        """
        self.run_args["cpu-affinity"] = ",".join(str(device) for device in devices)

    def set_gpu_affinity(self, devices: t.List[int]) -> None:
        """Set the GPU affinity for this job

        :param devices: list of GPU indices to execute on.
        """
        self.run_args["gpu-affinity"] = ",".join(str(device) for device in devices)
