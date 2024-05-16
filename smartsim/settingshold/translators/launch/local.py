from __future__ import annotations

from enum import Enum
import typing as t
from ..launchArgTranslator import LaunchArgTranslator
from ...launchCommand import LauncherType
from smartsim.log import get_logger
from ...common import IntegerArgument, StringArgument, FloatArgument                                                                       

logger = get_logger(__name__)

class LocalArgTranslator(LaunchArgTranslator):

    def launcher_str(self) -> str:
        """ Get the string representation of the launcher
        """
        return LauncherType.LocalLauncher.value

    def format_env_vars(self, env_vars: t.Dict[str, t.Optional[str]]) -> t.Union[t.List[str],None]:
        """Build environment variable string

        :returns: formatted list of strings to export variables
        """
        formatted = []
        for key, val in env_vars.items():
            if val is None:
                formatted.append(f"{key}=")
            else:
                formatted.append(f"{key}={val}")
        return formatted

    def format_launch_args(self, launch_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.Union[t.List[str],None]:
        """Build environment variable string

        :returns: formatted list of strings to export variables
        """
        formatted = []
        for arg, value in launch_args.items():
            formatted.append(arg)
            formatted.append(str(value))
        return formatted