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

import argparse
import collections
import importlib.metadata
import operator
import os
import re
import typing as t
from pathlib import Path

from tabulate import tabulate

from smartsim._core._cli.scripts.dragon_install import install_dragon
from smartsim._core._cli.utils import SMART_LOGGER_FORMAT, color_bool, pip
from smartsim._core._install import builder
from smartsim._core._install.buildenv import (
    BuildEnv,
    DbEngine,
    SetupError,
    Version_,
    VersionConflictError,
    Versioner,
)
from smartsim._core._install.builder import BuildError, Device
from smartsim._core._install.platform import Platform, Architecture, Device, OperatingSystem
from smartsim._core._install.mlpackages import DEFAULT_MLPACKAGES, PlatformPackages
from smartsim._core._install.redisaiBuilder import RedisAIBuilder
from smartsim._core.config import CONFIG
from smartsim._core.utils.helpers import installed_redisai_backends
from smartsim.error import SSConfigError
from smartsim.log import get_logger

logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)

# NOTE: all smartsim modules need full paths as the smart cli
#       may be installed into a different directory.

_TPinningStr = t.Literal["==", "!=", ">=", ">", "<=", "<", "~="]


def check_backends_install() -> bool:
    """Checks if backends have already been installed.
    Logs details on how to proceed forward
    if the RAI_PATH environment variable is set or if
    backends have already been installed.
    """
    rai_path = os.environ.get("RAI_PATH", "")
    installed = installed_redisai_backends()
    msg = ""

    if rai_path and installed:
        msg = (
            f"There is no need to build. backends are already built and "
            f"specified in the environment at 'RAI_PATH': {CONFIG.redisai}"
        )
    elif rai_path and not installed:
        msg = (
            "Before running 'smart build', unset your RAI_PATH environment "
            "variable with 'unset RAI_PATH'."
        )
    elif not rai_path and installed:
        msg = (
            "If you wish to re-run `smart build`, you must first run `smart clean`."
            " The following backend(s) must be removed: " + ", ".join(installed)
        )

    if msg:
        logger.error(msg)

    return not bool(msg)


def build_database(
    build_env: BuildEnv, versions: Versioner, keydb: bool, verbose: bool
) -> None:
    # check database installation
    database_name = "KeyDB" if keydb else "Redis"
    database_builder = builder.DatabaseBuilder(
        build_env(),
        jobs=build_env.JOBS,
        malloc=build_env.MALLOC,
        verbose=verbose,
    )
    if not database_builder.is_built:
        logger.info(
            f"Building {database_name} version {versions.REDIS} "
            f"from {versions.REDIS_URL}"
        )
        database_builder.build_from_git(versions.REDIS_URL, versions.REDIS_BRANCH)
        database_builder.cleanup()
    logger.info(f"{database_name} build complete!")


def build_redis_ai(
        platform: Platform,
        mlpackages: PlatformPackages,
        build_env: BuildEnv,
        verbose: bool
    ) -> None:
        logger.info("Building RedisAI and backends...")
        RAIBuilder = RedisAIBuilder(platform, mlpackages, build_env, verbose)
        RAIBuilder.build()
        RAIBuilder.cleanup_build()

def check_ml_python_packages(packages: PlatformPackages):
    def parse_requirement(requirement: str) -> t.Tuple[str, str, str]:
        operator_mappings = collections.defaultdict(ValueError("Invalid requirement operator"))
        operator_mappings.update({
            "==": operator.eq,
            "<=": operator.lte,
            ">=": operator.gte,
            "<": operator.lt,
            ">": operator.gt,
        })
        pattern = r"([a-zA-Z0-9_\-]+)([<>=!~]+)([\d\.]+)"
        match = re.match(pattern, requirement)
        if match:
            module_name, operator, version = match.groups()
            operator = operator_mappings[operator] if operator else None
            version = Version_(version) if version else None
            return module_name, operator, version
        else:
            raise ValueError(f"Invalid requirement string: {requirement}")

    missing = []
    conflicts = []

    for package in packages.values():
        for python_requirement in package.python_packages:
            module_name, operator, version = parse_requirement(python_requirement)
            try:
                dist = importlib.metadata.distribution(module_name)
                if operator and version:
                    if not operator(version, dist.version):
                        conflicts.append(f"{module_name} {version}")
            except importlib.metadata.PackageNotFoundError:
                missing.append(module_name)

    logger.warning(_format_incompatible_python_env_message(missing, conflicts))

def _format_incompatible_python_env_message(
    missing: t.Collection[str], conflicting: t.Collection[str]
) -> str:
    indent = "\n\t"
    fmt_list: t.Callable[[str, t.Collection[str]], str] = lambda n, l: (
        f"{n}:{indent}{indent.join(l)}" if l else ""
    )
    missing_str = fmt_list("Missing", missing)
    conflict_str = fmt_list("Conflicting", conflicting)
    sep = "\n" if missing_str and conflict_str else ""
    return (
        "Python Env Status Warning!\n"
        "Requested Packages are Missing or Conflicting:\n\n"
        f"{missing_str}{sep}{conflict_str}\n\n"
        "Consider installing packages at the requested versions via `pip` or "
        "uninstalling them, installing SmartSim with optional ML dependencies "
        "(`pip install smartsim[ml]`), and running `smart clean && smart build ...`"
    )


def _configure_keydb_build(versions: Versioner) -> None:
    """Configure the redis versions to be used during the build operation"""
    versions.REDIS = Version_("6.2.0")
    versions.REDIS_URL = "https://github.com/EQ-Alpha/KeyDB"
    versions.REDIS_BRANCH = "v6.2.0"

    CONFIG.conf_path = Path(CONFIG.core_path, "config", "keydb.conf")
    if not CONFIG.conf_path.resolve().is_file():
        raise SSConfigError(
            "Database configuration file at REDIS_CONF could not be found"
        )


# pylint: disable-next=too-many-statements
def execute(
    args: argparse.Namespace, _unparsed_args: t.Optional[t.List[str]] = None, /
) -> int:

    # Unpack various arguments
    verbose = args.v
    keydb = args.keydb
    device = Device.from_str(args.device.lower())
    is_dragon_requested = args.dragon

    # The user should never have to specify the OS and Architecture
    current_platform = Platform(
        OperatingSystem.autodetect(),
        Architecture.autodetect(),
        device
    )

    # Configure the ML Packages
    mlpackages = DEFAULT_MLPACKAGES[current_platform]
    if args.libtorch_dir:
        mlpackages["libtorch"].set_lib_source(args.libtorch_dir)
    if args.libtensorflow_dir:
        mlpackages["libtensorflow"].set_lib_source(args.libtensorflow_dir)
    if args.onnxruntime_dir:
        mlpackages["onnxruntime"].set_lib_source(args.onnxruntime_dir)

    # Build all backends by default, pop off the ones that user wants skipped
    if args.skip_torch:
        mlpackages.pop("libtorch")
    if args.skip_tensorflow:
        mlpackages.pop("libtensorflow")
    if args.skip_onnx:
        mlpackages.pop("onnxruntime")

    build_env = BuildEnv(checks=True)
    logger.info("Running SmartSim build process...")

    logger.info("Checking requested versions...")
    versions = Versioner()

    logger.debug("Checking for build tools...")

    if verbose:
        logger.info("Build Environment:")
        env = build_env.as_dict()
        env_vars = list(env.keys())
        print(tabulate(env, headers=env_vars, tablefmt="github"), "\n")

    if keydb:
        _configure_keydb_build(versions)

    if verbose:
        db_name: DbEngine = "KEYDB" if keydb else "REDIS"
        logger.info("Version Information:")
        vers = versions.as_dict(db_name=db_name)
        version_names = list(vers.keys())
        print(tabulate(vers, headers=version_names, tablefmt="github"), "\n")

    if is_dragon_requested:
        install_to = CONFIG.core_path / ".dragon"
        return_code = install_dragon(install_to)

        if return_code == 0:
            logger.info("Dragon installation complete")
        elif return_code == 1:
            logger.info("Dragon installation not supported on platform")
        else:
            logger.warning("Dragon installation failed")

    # REDIS/KeyDB
    build_database(build_env, versions, keydb, verbose)
    build_redis_ai(current_platform, mlpackages, build_env, verbose)

    backends = installed_redisai_backends()
    backends_str = ", ".join(s.capitalize() for s in backends) if backends else "No"
    logger.info(f"{backends_str} backend(s) built")
    check_ml_python_packages(mlpackages)

    logger.info("SmartSim build complete!")
    return os.EX_OK


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Builds the parser for the command"""
    warn_usage = "(ONLY USE IF NEEDED)"
    parser.add_argument(
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose build process",
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        default=Device.CPU.value,
        choices=[*(device.value for device in Device), "gpu"],
        help="Device to build ML runtimes for",
    )
    parser.add_argument(
        "--dragon",
        action="store_true",
        default=False,
        help="Install the dragon runtime",
    )
    parser.add_argument(
        "--with-python-packages",
        action="store_true",
        help="Install the python packages that match the backends",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Do not build PyTorch backend",
    )
    parser.add_argument(
        "--skip-tensorflow",
        action="store_true",
        help="Do not build TensorFlow backend",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Build ONNX backend (off by default)",
    )
    parser.add_argument(
        "--libtorch-dir",
        default=None,
        type=str,
        help=f"Path to custom libtorch directory{warn_usage}",
    )
    parser.add_argument(
        "--libtensorflow-dir",
        default=None,
        type=str,
        help=f"Path to custom libtensorflow directory {warn_usage}",
    )
    parser.add_argument(
        "--onnxruntime-dir",
        default=None,
        type=str,
        help=f"Path to onnxruntime libtensorflow directory {warn_usage}",
    )
    parser.add_argument(
        "--keydb",
        action="store_true",
        default=False,
        help="Build KeyDB instead of Redis",
    )