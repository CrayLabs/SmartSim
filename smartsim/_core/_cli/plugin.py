import argparse
import shutil
import subprocess as sp
import typing as t

from smartsim._core._cli.utils import MenuItemConfig


plugins: t.List[t.Callable[[], MenuItemConfig]] = []


def dynamic_execute(cmd: str) -> t.Callable[[argparse.Namespace, t.List[str]], int]:
    def process_execute(_args: argparse.Namespace, unparsed_args: t.List[str]) -> int:
        if not shutil.which(cmd):
            raise ValueError(f"{cmd} plugin not found. Please ensure it is installed")
        combined_cmd = [cmd] + unparsed_args
        with sp.Popen(combined_cmd, stdout=sp.PIPE, stderr=sp.PIPE) as process:
            stdout, _ = process.communicate()
            while process.returncode is None:
                stdout, _ = process.communicate()
                print(stdout.decode("utf-8"))

            return process.returncode

    return process_execute


def dashboard() -> MenuItemConfig:
    return MenuItemConfig(
        "dashboard",
        "Start the SmartSim dashboard",
        dynamic_execute("smart-dash"),
        is_plugin=True,
    )


plugins.append(dashboard)
