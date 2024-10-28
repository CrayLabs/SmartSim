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

from smartsim.log import get_logger

from ...common import set_check_input
from ...launch_command import LauncherType
from ..launch_arguments import LaunchArguments

logger = get_logger(__name__)


class DragonLaunchArguments(LaunchArguments):
    def launcher_str(self) -> str:
        """Get the string representation of the launcher

        :returns: The string representation of the launcher
        """
        return LauncherType.Dragon.value

    def set_nodes(self, nodes: int) -> None:
        """Set the number of nodes

        :param nodes: number of nodes to run with
        """
        self.set("nodes", str(nodes))

    def set_tasks_per_node(self, tasks_per_node: int) -> None:
        """Set the number of tasks for this job

        :param tasks_per_node: number of tasks per node
        """
        self.set("tasks_per_node", str(tasks_per_node))

    @override
    def set(self, key: str, value: str | None) -> None:
        """Set an arbitrary launch argument

        :param key: The launch argument
        :param value: A string representation of the value for the launch
            argument (if applicable), otherwise `None`
        """
        set_check_input(key, value)
        if key in self._launch_args and key != self._launch_args[key]:
            logger.warning(f"Overwritting argument '{key}' with value '{value}'")
        self._launch_args[key] = value

    def set_node_feature(self, feature_list: t.Union[str, t.List[str]]) -> None:
        """Specify the node feature for this job

        :param feature_list: a collection of strings representing the required
         node features. Currently supported node features are: "gpu"
        """
        if isinstance(feature_list, str):
            feature_list = feature_list.strip().split()
        elif not all(isinstance(feature, str) for feature in feature_list):
            raise TypeError("feature_list must be string or list of strings")
        self.set("node-feature", ",".join(feature_list))

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :raises ValueError: if an empty host list is supplied
        """
        if not host_list:
            raise ValueError("empty hostlist provided")

        if isinstance(host_list, str):
            host_list = host_list.replace(" ", "").split(",")

        # strip out all whitespace-only values
        cleaned_list = [host.strip() for host in host_list if host and host.strip()]
        if not len(cleaned_list) == len(host_list):
            raise ValueError(f"invalid names found in hostlist: {host_list}")
        self.set("host-list", ",".join(cleaned_list))

    def set_cpu_affinity(self, devices: t.List[int]) -> None:
        """Set the CPU affinity for this job

        :param devices: list of CPU indices to execute on
        """
        self.set("cpu-affinity", ",".join(str(device) for device in devices))

    def set_gpu_affinity(self, devices: t.List[int]) -> None:
        """Set the GPU affinity for this job

        :param devices: list of GPU indices to execute on.
        """
        self.set("gpu-affinity", ",".join(str(device) for device in devices))
