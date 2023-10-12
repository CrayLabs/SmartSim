import argparse
import importlib.metadata
import pathlib
import typing as t

from tabulate import tabulate

import smartsim._core._cli.utils as _utils
import smartsim._core.utils.helpers as _helpers
from smartsim._core._install.buildenv import BuildEnv as _BuildEnv

_MISSING_DEP = _helpers.colorize("Not Installed", "red")


def execute(_args: argparse.Namespace, /) -> int:
    print("\nSmart Python Packages:")
    print(
        tabulate(
            [
                ["SmartSim", _fmt_py_pkg_version("smartsim")],
                ["SmartRedis", _fmt_py_pkg_version("smartredis")],
            ],
            headers=["Name", "Version"],
            tablefmt="fancy_outline",
        ),
        end="\n\n",
    )

    print("Orchestrator Configuration:")
    db_path = _utils.get_db_path()
    db_table = [["Installed", _fmt_installed_db(db_path)]]
    if db_path:
        db_table.append(["Location", str(db_path)])
    print(tabulate(db_table, tablefmt="fancy_outline"), end="\n\n")

    print("Redis AI Configuration:")
    rai_path = _helpers.redis_install_base().parent / "redisai.so"
    rai_table = [["Status", _fmt_installed_redis_ai(rai_path)]]
    if rai_path.is_file():
        rai_table.append(["Location", str(rai_path)])
    print(tabulate(rai_table, tablefmt="fancy_outline"), end="\n\n")

    print("Machine Learning Backends:")
    backends = _helpers.installed_redisai_backends()
    print(
        tabulate(
            [
                [
                    "Tensorflow",
                    _utils.color_bool("tensorflow" in backends),
                    _fmt_py_pkg_version("tensorflow"),
                ],
                [
                    "Torch",
                    _utils.color_bool("torch" in backends),
                    _fmt_py_pkg_version("torch"),
                ],
                [
                    "ONNX",
                    _utils.color_bool("onnxruntime" in backends),
                    _fmt_py_pkg_version("onnx"),
                ],
            ],
            headers=["Name", "Backend Available", "Python Package"],
            tablefmt="fancy_outline",
        ),
        end="\n\n",
    )
    return 0


def _fmt_installed_db(db_path: t.Optional[pathlib.Path]) -> str:
    if db_path is None:
        return _MISSING_DEP
    db_name, _ = db_path.name.split("-", 1)
    return _helpers.colorize(db_name.upper(), "green")


def _fmt_installed_redis_ai(rai_path: pathlib.Path) -> str:
    if not rai_path.is_file():
        return _MISSING_DEP
    return _helpers.colorize("Installed", "green")


def _fmt_py_pkg_version(pkg_name: str) -> str:
    try:
        return _helpers.colorize(_BuildEnv.get_py_package_version(pkg_name), "green")
    except importlib.metadata.PackageNotFoundError:
        return _MISSING_DEP
