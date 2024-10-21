import argparse
import importlib.metadata
import os
import pathlib
import typing as t

from tabulate import tabulate

import smartsim._core.utils.helpers as _helpers
from smartsim._core._install.buildenv import BuildEnv as _BuildEnv

_MISSING_DEP = _helpers.colorize("Not Installed", "red")


def execute(
    _args: argparse.Namespace, _unparsed_args: t.Optional[t.List[str]] = None, /
) -> int:
    print("\nSmart Python Packages:")
    print(
        tabulate(
            [
                ["SmartSim", _fmt_py_pkg_version("smartsim")],
            ],
            headers=["Name", "Version"],
            tablefmt="fancy_outline",
        ),
        end="\n\n",
    )

    print("Dragon Installation:")
    # TODO: Fix hardcoded dragon version
    dragon_version = "0.10"

    fs_table = [["Version", str(dragon_version)]]
    print(tabulate(fs_table, tablefmt="fancy_outline"), end="\n\n")

    print("Machine Learning Packages:")
    print(
        tabulate(
            [
                [
                    "Tensorflow",
                    _fmt_py_pkg_version("tensorflow"),
                ],
                [
                    "Torch",
                    _fmt_py_pkg_version("torch"),
                ],
                [
                    "ONNX",
                    _fmt_py_pkg_version("onnx"),
                ],
            ],
            headers=["Name", "Python Package"],
            tablefmt="fancy_outline",
        ),
        end="\n\n",
    )
    return os.EX_OK


def _fmt_installed_fs(fs_path: t.Optional[pathlib.Path]) -> str:
    if fs_path is None:
        return _MISSING_DEP
    fs_name, _ = fs_path.name.split("-", 1)
    return _helpers.colorize(fs_name.upper(), "green")


def _fmt_py_pkg_version(pkg_name: str) -> str:
    try:
        return _helpers.colorize(_BuildEnv.get_py_package_version(pkg_name), "green")
    except importlib.metadata.PackageNotFoundError:
        return _MISSING_DEP
