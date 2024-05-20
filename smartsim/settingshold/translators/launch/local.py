from __future__ import annotations

import typing as t
from ..launchArgTranslator import LaunchArgTranslator
from ...launchCommand import LauncherType
from smartsim.log import get_logger
from ...common import StringArgument                                                                     

logger = get_logger(__name__)

class LocalArgTranslator(LaunchArgTranslator):

    def launcher_str(self) -> str:
        """ Get the string representation of the launcher
        """
        return LauncherType.LocalLauncher.value

    def _set_reserved_launch_args(self) -> set[str]:
        return set()

    def format_env_vars(self, env_vars: StringArgument) -> t.Union[t.List[str],None]:
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

    def format_launcher_args(self, launcher_args: t.Dict[str, t.Union[str,int,float,None]]) -> t.Union[t.List[str],None]:
        """Build launcher argument string

        :returns: formatted list of launcher arguments
        """
        formatted = []
        for arg, value in launcher_args.items():
            formatted.append(arg)
            formatted.append(str(value))
        return formatted