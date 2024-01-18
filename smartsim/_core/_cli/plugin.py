import argparse
import importlib.util
import os
import subprocess as sp
import sys
import typing as t

import smartsim.log
from smartsim._core._cli.utils import SMART_LOGGER_FORMAT, MenuItemConfig
from smartsim.error.errors import SmartSimCLIActionCancelled

_LOGGER = smartsim.log.get_logger("Smart", fmt=SMART_LOGGER_FORMAT)


def dynamic_execute(
    cmd: str, plugin_name: str
) -> t.Callable[[argparse.Namespace, t.List[str]], int]:
    def process_execute(
        _args: argparse.Namespace, unparsed_args: t.List[str], /
    ) -> int:
        try:
            spec = importlib.util.find_spec(cmd)
            if spec is None:
                raise AttributeError
        except (ModuleNotFoundError, AttributeError):
            _LOGGER.error(f"{cmd} plugin not found. Please ensure it is installed")
            return os.EX_CONFIG

        combined_cmd = [sys.executable, "-m", cmd] + unparsed_args

        try:
            completed_proc = sp.run(combined_cmd, check=False)
        except KeyboardInterrupt as ex:
            msg = f"{plugin_name} terminated by user"
            raise SmartSimCLIActionCancelled(msg) from ex
        return completed_proc.returncode

    return process_execute


def dashboard() -> MenuItemConfig:
    return MenuItemConfig(
        "dashboard",
        (
            "Start the SmartSim dashboard to monitor experiment output from a "
            "graphical user interface. This requires that the SmartSim Dashboard "
            "Package be installed. For more infromation please visit "
            "https://github.com/CrayLabs/SmartDashboard"
        ),
        dynamic_execute("smartdashboard", "Dashboard"),
        is_plugin=True,
    )


plugins = (dashboard,)
