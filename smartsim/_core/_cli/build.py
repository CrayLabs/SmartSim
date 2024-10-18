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
import importlib.metadata
import operator
import os
import re
import shutil
import textwrap
import typing as t
from pathlib import Path

from tabulate import tabulate

from smartsim._core._cli.scripts.dragon_install import (
    DEFAULT_DRAGON_REPO,
    DEFAULT_DRAGON_VERSION,
    DragonInstallRequest,
    display_post_install_logs,
    install_dragon,
)
from smartsim._core._cli.utils import SMART_LOGGER_FORMAT
from smartsim._core._install import builder
from smartsim._core._install.buildenv import BuildEnv, DbEngine, Version_, Versioner
from smartsim._core._install.mlpackages import (
    DEFAULT_MLPACKAGE_PATH,
    DEFAULT_MLPACKAGES,
    MLPackageCollection,
    load_platform_configs,
)
from smartsim._core._install.platform import (
    Architecture,
    Device,
    OperatingSystem,
    Platform,
)
from smartsim._core._install.redisaiBuilder import RedisAIBuilder
from smartsim._core.config import CONFIG
from smartsim._core.utils.helpers import installed_redisai_backends
from smartsim.error import SSConfigError
from smartsim.log import get_logger

logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)

# NOTE: all smartsim modules need full paths as the smart cli
#       may be installed into a different directory.


def check_backends_install() -> bool:
    """Checks if backends have already been installed.
    Logs details on how to proceed forward
    if the SMARTSIM_RAI_LIB environment variable is set or if
    backends have already been installed.
    """
    rai_path = os.environ.get("SMARTSIM_RAI_LIB", "")
    installed = installed_redisai_backends()
    msg = ""

    if rai_path and installed:
        msg = (
            f"There is no need to build. backends are already built and "
            f"specified in the environment at 'SMARTSIM_RAI_LIB': {CONFIG.redisai}"
        )
    elif rai_path and not installed:
        msg = (
            "Before running 'smart build', unset your SMARTSIM_RAI_LIB environment "
            "variable with 'unset SMARTSIM_RAI_LIB'."
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
        database_builder.build_from_git(
            versions.REDIS_URL, branch=versions.REDIS_BRANCH
        )
        database_builder.cleanup()
        logger.info(f"{database_name} build complete!")
    else:
        logger.warning(
            f"{database_name} was previously built, run 'smart clobber' to rebuild"
        )


def build_redis_ai(
    platform: Platform,
    mlpackages: MLPackageCollection,
    build_env: BuildEnv,
    verbose: bool,
) -> None:
    logger.info("Building RedisAI and backends...")
    rai_builder = RedisAIBuilder(
        platform, mlpackages, build_env, CONFIG.build_path, verbose
    )
    rai_builder.build()
    rai_builder.cleanup_build()


def parse_requirement(
    requirement: str,
) -> t.Tuple[str, t.Optional[str], t.Callable[[Version_], bool]]:
    operators = {
        "==": operator.eq,
        "<=": operator.le,
        ">=": operator.ge,
        "<": operator.lt,
        ">": operator.gt,
    }
    semantic_version_pattern = r"\d+(?:\.\d+(?:\.\d+)?)?([^\s]*)"
    pattern = (
        r"^"  # Start
        r"([a-zA-Z0-9_\-]+)"  # Package name
        r"(?:\[[a-zA-Z0-9_\-,]+\])?"  # Any extras
        r"(?:([<>=!~]{1,2})"  # Pinning string
        rf"({semantic_version_pattern}))?"  # A version number
        r"$"  # End
    )
    match = re.match(pattern, requirement)
    if match is None:
        raise ValueError(f"Invalid requirement string: {requirement}")
    module_name, cmp_op, version_str, suffix = match.groups()
    version = Version_(version_str) if version_str is not None else None
    if cmp_op is None:
        is_compatible = lambda _: True  # pylint: disable=unnecessary-lambda-assignment
    elif (cmp := operators.get(cmp_op, None)) is None:
        raise ValueError(f"Unrecognized comparison operator: {cmp_op}")
    else:

        def is_compatible(other: Version_) -> bool:
            assert version is not None  # For type check, always should be true
            match_ = re.match(rf"^{semantic_version_pattern}$", other)
            return (
                cmp(other, version) and match_ is not None and match_.group(1) == suffix
            )

    return module_name, f"{cmp_op}{version}" if version else None, is_compatible


def check_ml_python_packages(packages: MLPackageCollection) -> None:
    missing = []
    conflicts = []

    for package in packages.values():
        for requirement in package.python_packages:
            module_name, version_spec, is_compatible = parse_requirement(requirement)
            try:
                installed = BuildEnv.get_py_package_version(module_name)
                if not is_compatible(installed):
                    conflicts.append(
                        f"{module_name}: {installed} is installed, "
                        f"but {version_spec or 'Any'} is required"
                    )
            except importlib.metadata.PackageNotFoundError:
                missing.append(module_name)

    if missing or conflicts:
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

    return textwrap.dedent(f"""\
        Python Package Warning:

        Requested packages are missing or have a version mismatch with
        their respective backend:

        {missing_str}{sep}{conflict_str}

        Consider uninstalling any conflicting packages and rerunning
        `smart build` if you encounter issues.
        """)


def _configure_keydb_build(versions: Versioner) -> None:
    """Configure the redis versions to be used during the build operation"""
    versions.REDIS = Version_("6.2.0")
    versions.REDIS_URL = "https://github.com/EQ-Alpha/KeyDB.git"
    versions.REDIS_BRANCH = "v6.2.0"

    CONFIG.conf_path = Path(CONFIG.core_path, "config", "keydb.conf")
    if not CONFIG.conf_path.resolve().is_file():
        raise SSConfigError(
            "Database configuration file at SMARTSIM_REDIS_CONF could not be found"
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
    dragon_repo = args.dragon_repo
    dragon_version = args.dragon_version

    if Path(CONFIG.build_path).exists():
        logger.warning(f"Build path already exists, removing: {CONFIG.build_path}")
        shutil.rmtree(CONFIG.build_path)

    # The user should never have to specify the OS and Architecture
    current_platform = Platform(
        OperatingSystem.autodetect(), Architecture.autodetect(), device
    )

    # Configure the ML Packages
    configs = load_platform_configs(Path(args.config_dir))
    mlpackages = configs[current_platform]

    # Build all backends by default, pop off the ones that user wants skipped
    if args.skip_torch and "libtorch" in mlpackages:
        mlpackages.pop("libtorch")
    if args.skip_tensorflow and "libtensorflow" in mlpackages:
        mlpackages.pop("libtensorflow")
    if args.skip_onnx and "onnxruntime" in mlpackages:
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

    logger.info("ML Packages")
    print(mlpackages)

    if is_dragon_requested or dragon_repo or dragon_version:
        install_to = CONFIG.core_path / ".dragon"

        try:
            request = DragonInstallRequest(
                install_to,
                dragon_repo,
                dragon_version,
            )
            return_code = install_dragon(request)
        except ValueError as ex:
            return_code = 2
            logger.error(" ".join(ex.args))

        if return_code == 0:
            display_post_install_logs()

        elif return_code == 1:
            logger.info("Dragon installation not supported on platform")
        else:
            logger.warning("Dragon installation failed")

    # REDIS/KeyDB
    build_database(build_env, versions, keydb, verbose)

    if (CONFIG.lib_path / "redisai.so").exists():
        logger.warning("RedisAI was previously built, run 'smart clean' to rebuild")
    elif not args.skip_backends:
        build_redis_ai(current_platform, mlpackages, build_env, verbose)
    else:
        logger.info("Skipping compilation of RedisAI and backends")

    backends = installed_redisai_backends()
    backends_str = ", ".join(s.capitalize() for s in backends) if backends else "No"
    logger.info(f"{backends_str} backend(s) available")

    if not args.skip_python_packages:
        for package in mlpackages.values():
            logger.info(f"Installing python packages for {package.name}")
            package.pip_install(quiet=not verbose)
    check_ml_python_packages(mlpackages)

    logger.info("SmartSim build complete!")
    return os.EX_OK


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Builds the parser for the command"""

    available_devices = []
    for platform in DEFAULT_MLPACKAGES:
        if (platform.operating_system == OperatingSystem.autodetect()) and (
            platform.architecture == Architecture.autodetect()
        ):
            available_devices.append(platform.device.value)

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
        choices=available_devices,
        help="Device to build ML runtimes for",
    )
    parser.add_argument(
        "--dragon",
        action="store_true",
        default=False,
        help="Install the dragon runtime",
    )
    parser.add_argument(
        "--dragon-repo",
        default=None,
        type=str,
        help=(
            "Specify a git repo containing dragon release assets "
            f"(e.g. {DEFAULT_DRAGON_REPO})"
        ),
    )
    parser.add_argument(
        "--dragon-version",
        default=None,
        type=str,
        help=f"Specify the dragon version to install (e.g. {DEFAULT_DRAGON_VERSION})",
    )
    parser.add_argument(
        "--skip-python-packages",
        action="store_true",
        help="Do not install the python packages that match the backends",
    )
    parser.add_argument(
        "--skip-backends",
        action="store_true",
        help="Do not compile RedisAI and the backends",
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
        help="Do not build the ONNX backend",
    )
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_MLPACKAGE_PATH),
        type=str,
        help="Path to directory with JSON files describing platform and packages",
    )
    parser.add_argument(
        "--keydb",
        action="store_true",
        default=False,
        help="Build KeyDB instead of Redis",
    )
