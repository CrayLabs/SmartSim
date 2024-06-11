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

from smartsim.log import get_logger

from ...common import StringArgument, set_check_input
from ...launchCommand import LauncherType
from ..launchArgBuilder import LaunchArgBuilder

logger = get_logger(__name__)


class DragonArgBuilder(LaunchArgBuilder):
    def launcher_str(self) -> str:
        """Get the string representation of the launcher"""
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
        self.set("tasks-per-node", str(tasks_per_node))

    def set(self, key: str, value: str | None) -> None:
        """Set the launch arguments"""
        set_check_input(key, value)
        if key in self._launch_args and key != self._launch_args[key]:
            logger.warning(f"Overwritting argument '{key}' with value '{value}'")
        self._launch_args[key] = value
