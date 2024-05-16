from __future__ import annotations

from enum import Enum
import typing as t
from ..launchArgTranslator import LaunchArgTranslator
from ...common import IntegerArgument, StringArgument, FloatArgument
from ...launchCommand import LauncherType
from smartsim.log import get_logger                                                                                

logger = get_logger(__name__)

class DragonArgTranslator(LaunchArgTranslator):

    def launcher_str(self) -> str:
        """ Get the string representation of the launcher
        """
        return LauncherType.DragonLauncher.value

    def set_nodes(self, nodes: int) -> t.Union[IntegerArgument, None]:
        """Set the number of nodes

        :param nodes: number of nodes to run with
        """
        return {"nodes": nodes}

    def set_tasks_per_node(self, tasks_per_node: int) -> t.Union[IntegerArgument, None]:
        """Set the number of tasks for this job

        :param tasks_per_node: number of tasks per node
        """
        return {"tasks-per-node": tasks_per_node}