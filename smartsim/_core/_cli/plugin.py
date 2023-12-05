import argparse
import importlib
import os
import sys
import subprocess as sp
import typing as t

from smartsim._core._cli.utils import MenuItemConfig
from smartsim.error.errors import SmartSimCLIActionCancelled


def dynamic_execute(
    cmd: str, plugin_name: str
) -> t.Callable[[argparse.Namespace, t.List[str]], int]:
    def process_execute(
        _args: argparse.Namespace, unparsed_args: t.List[str], /
    ) -> int:
        not_found = f"{cmd} plugin not found. Please ensure it is installed"
        try:
            spec = importlib.util.find_spec(cmd)
            if spec is None:
                raise AttributeError()
        except (ModuleNotFoundError, AttributeError):
            print(not_found)
            return os.EX_CONFIG

        combined_cmd = [sys.executable, "-m", cmd] + unparsed_args

        try:
            with sp.Popen(combined_cmd, stdout=sp.PIPE, stderr=sp.PIPE) as process:
                stdout, _ = process.communicate()
                while process.returncode is None:
                    stdout, _ = process.communicate()

                plugin_stdout = stdout.decode("utf-8")
                print(plugin_stdout)
                return process.returncode
        except KeyboardInterrupt as ex:
            msg = f"{plugin_name} terminated by user"
            raise SmartSimCLIActionCancelled(msg) from ex

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
        dynamic_execute("smartdashboard.Experiment_Overview", "Dashboard"),
        is_plugin=True,
    )


plugins = (dashboard,)
