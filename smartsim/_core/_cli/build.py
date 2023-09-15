# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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
import os
import sys
import typing as t
from pathlib import Path

from tabulate import tabulate

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
from smartsim._core._install.builder import BuildError
from smartsim._core.config import CONFIG
from smartsim._core.utils.helpers import installed_redisai_backends
from smartsim.error import SSConfigError
from smartsim.log import get_logger

logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)

# NOTE: all smartsim modules need full paths as the smart cli
#       may be installed into a different directory.


_TDeviceStr = t.Literal["cpu", "gpu"]
_TPinningStr = t.Literal["==", "!=", ">=", ">", "<=", "<", "~="]


def check_py_onnx_version(versions: Versioner) -> None:
    """Check Python environment for ONNX installation"""
    if not versions.ONNX:
        py_version = sys.version_info
        msg = (
            "An onnx wheel is not available for "
            f"Python {py_version.major}.{py_version.minor}. "
            "Instead consider using Python 3.8 or 3.9 with Onnx "
        )
        if sys.platform == "linux":
            msg += "1.2.5 or "
        msg += "1.2.7."
        raise SetupError(msg)
    _check_packages_in_python_env(
        {
            "onnx": Version_(versions.ONNX),
            "skl2onnx": Version_(versions.REDISAI.skl2onnx),
            "onnxmltools": Version_(versions.REDISAI.onnxmltools),
            "scikit-learn": Version_(getattr(versions.REDISAI, "scikit-learn")),
        },
    )


def check_py_tf_version(versions: Versioner) -> None:
    """Check Python environment for TensorFlow installation"""
    _check_packages_in_python_env({"tensorflow": Version_(versions.TENSORFLOW)})


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
        build_env(), build_env.MALLOC, build_env.JOBS, verbose
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
    build_env: BuildEnv,
    versions: Versioner,
    device: _TDeviceStr,
    use_torch: bool = True,
    use_tf: bool = True,
    use_onnx: bool = False,
    torch_dir: t.Union[str, Path, None] = None,
    libtf_dir: t.Union[str, Path, None] = None,
    verbose: bool = False,
) -> None:
    # make sure user isn't trying to do something silly on MacOS
    if build_env.PLATFORM == "darwin" and device == "gpu":
        raise BuildError("SmartSim does not support GPU on MacOS")

    # decide which runtimes to build
    print("\nML Backends Requested")
    backends_table = [
        ["PyTorch", versions.TORCH, color_bool(use_torch)],
        ["TensorFlow", versions.TENSORFLOW, color_bool(use_tf)],
        ["ONNX", versions.ONNX or "Unavailable", color_bool(use_onnx)],
    ]
    print(tabulate(backends_table, tablefmt="fancy_outline"), end="\n\n")
    print(f"Building for GPU support: {color_bool(device == 'gpu')}\n")

    if not check_backends_install():
        sys.exit(1)

    # TORCH
    if use_torch and torch_dir:
        torch_dir = Path(torch_dir).resolve()
        if not torch_dir.is_dir():
            raise SetupError(
                f"Could not find requested user Torch installation: {torch_dir}"
            )

    # TF
    if use_tf and libtf_dir:
        libtf_dir = Path(libtf_dir).resolve()
        if not libtf_dir.is_dir():
            raise SetupError(
                f"Could not find requested user TF installation: {libtf_dir}"
            )

    build_env_dict = build_env()

    rai_builder = builder.RedisAIBuilder(
        build_env=build_env_dict,
        torch_dir=str(torch_dir) if torch_dir else "",
        libtf_dir=str(libtf_dir) if libtf_dir else "",
        build_torch=use_torch,
        build_tf=use_tf,
        build_onnx=use_onnx,
        jobs=build_env.JOBS,
        verbose=verbose,
    )

    if rai_builder.is_built:
        logger.info("RedisAI installed. Run `smart clean` to remove.")
    else:
        # get the build environment, update with CUDNN env vars
        # if present and building for GPU, otherwise warn the user
        if device == "gpu":
            gpu_env = build_env.get_cudnn_env()
            cudnn_env_vars = [
                "CUDNN_LIBRARY",
                "CUDNN_INCLUDE_DIR",
                "CUDNN_INCLUDE_PATH",
                "CUDNN_LIBRARY_PATH",
            ]
            if not gpu_env:
                logger.warning(
                    "CUDNN environment variables not found.\n"
                    f"Looked for {cudnn_env_vars}"
                )
            else:
                build_env_dict.update(gpu_env)
        # update RAI build env with cudnn env vars
        rai_builder.env = build_env_dict

        logger.info(
            f"Building RedisAI version {versions.REDISAI}"
            f" from {versions.REDISAI_URL}"
        )

        # NOTE: have the option to add other builds here in the future
        # like "from_tarball"
        rai_builder.build_from_git(
            versions.REDISAI_URL, versions.REDISAI_BRANCH, device
        )
        logger.info("ML Backends and RedisAI build complete!")


def check_py_torch_version(versions: Versioner, device: _TDeviceStr = "cpu") -> None:
    """Check Python environment for TensorFlow installation"""

    if BuildEnv.is_macos():
        if device == "gpu":
            raise BuildError("SmartSim does not support GPU on MacOS")
        device_suffix = ""
    else:  # linux
        if device == "cpu":
            device_suffix = versions.TORCH_CPU_SUFFIX
        elif device == "gpu":
            device_suffix = versions.TORCH_CUDA_SUFFIX
        else:
            raise BuildError("Unrecognized device requested")

    torch_deps = {
        "torch": Version_(f"{versions.TORCH}{device_suffix}"),
        "torchvision": Version_(f"{versions.TORCHVISION}{device_suffix}"),
    }
    missing, conflicts = _assess_python_env(
        torch_deps,
        package_pinning="==",
        validate_installed_version=_create_torch_version_validator(
            with_suffix=device_suffix
        ),
    )

    if len(missing) == len(torch_deps) and not conflicts:
        # All PyTorch deps are not installed and there are no conflicting
        # python packages. We can try to install torch deps into the current env.
        logger.info(
            "Torch version not found in python environment. "
            "Attempting to install via `pip`"
        )
        pip(
            "install",
            "-f",
            "https://download.pytorch.org/whl/torch_stable.html",
            *(f"{package}=={version}" for package, version in torch_deps.items()),
        )
    elif missing or conflicts:
        logger.warning(_format_incompatible_python_env_message(missing, conflicts))


def _create_torch_version_validator(
    with_suffix: str,
) -> t.Callable[[str, t.Optional[Version_]], bool]:
    def check_torch_version(package: str, version: t.Optional[Version_]) -> bool:
        if not BuildEnv.check_installed(package, version):
            return False
        # Default check only looks at major/minor version numbers,
        # Torch requires we look at the patch as well
        installed = BuildEnv.get_py_package_version(package)
        if with_suffix and with_suffix not in installed.patch:
            raise VersionConflictError(
                package,
                installed,
                version or Version_(f"X.X.X{with_suffix}"),
                msg=(
                    f"{package}=={installed} does not satisfy device "
                    f"suffix requirement: {with_suffix}"
                ),
            )
        return True

    return check_torch_version


def _check_packages_in_python_env(
    packages: t.Mapping[str, t.Optional[Version_]],
    package_pinning: _TPinningStr = "==",
    validate_installed_version: t.Optional[
        t.Callable[[str, t.Optional[Version_]], bool]
    ] = None,
) -> None:
    # TODO: Do not like how the default validation function will always look for
    #       a `==` pinning. Maybe turn `BuildEnv.check_installed` into a factory
    #       that takes a pinning and returns an appropriate validation fn?
    validate_installed_version = validate_installed_version or BuildEnv.check_installed
    missing, conflicts = _assess_python_env(
        packages,
        package_pinning,
        validate_installed_version,
    )

    if missing or conflicts:
        logger.warning(_format_incompatible_python_env_message(missing, conflicts))


def _assess_python_env(
    packages: t.Mapping[str, t.Optional[Version_]],
    package_pinning: _TPinningStr,
    validate_installed_version: t.Callable[[str, t.Optional[Version_]], bool],
) -> t.Tuple[t.List[str], t.List[str]]:
    missing: t.List[str] = []
    conflicts: t.List[str] = []

    for name, version in packages.items():
        spec = f"{name}{package_pinning}{version}" if version else name
        try:
            if not validate_installed_version(name, version):
                # Not installed!
                missing.append(spec)
        except VersionConflictError:
            # Incompatible version found
            conflicts.append(spec)

    return missing, conflicts


def _format_incompatible_python_env_message(
    missing: t.Iterable[str], conflicting: t.Iterable[str]
) -> str:
    indent = "\n\t"
    fmt_list: t.Callable[[str, t.Iterable[str]], str] = (
        lambda n, l: f"{n}:{indent}{indent.join(l)}" if l else ""
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


def execute(args: argparse.Namespace) -> int:
    verbose = args.v
    keydb = args.keydb
    device: _TDeviceStr = args.device

    # torch and tf build by default
    pt = not args.no_pt  # pylint: disable=invalid-name
    tf = not args.no_tf  # pylint: disable=invalid-name
    onnx = args.onnx

    build_env = BuildEnv(checks=True)
    logger.info("Running SmartSim build process...")

    logger.info("Checking requested versions...")
    versions = Versioner()

    logger.info("Checking for build tools...")

    if verbose:
        logger.info("Build Environment:")
        env = build_env.as_dict()
        env_vars = list(env.keys())
        print(tabulate(env, headers=env_vars, tablefmt="github"), "\n")

    if keydb:
        versions.REDIS = Version_("6.2.0")
        versions.REDIS_URL = "https://github.com/EQ-Alpha/KeyDB"
        versions.REDIS_BRANCH = "v6.2.0"
        CONFIG.conf_path = Path(CONFIG.core_path, "config", "keydb.conf")
        if not CONFIG.conf_path.resolve().is_file():
            raise SSConfigError(
                "Database configuration file at REDIS_CONF could not be found"
            )

    if verbose:
        db_name: DbEngine = "KEYDB" if keydb else "REDIS"
        logger.info("Version Information:")
        vers = versions.as_dict(db_name=db_name)
        version_names = list(vers.keys())
        print(tabulate(vers, headers=version_names, tablefmt="github"), "\n")

    try:
        if not args.only_python_packages:
            # REDIS/KeyDB
            build_database(build_env, versions, keydb, verbose)

            # REDISAI
            build_redis_ai(
                build_env,
                versions,
                device,
                pt,
                tf,
                onnx,
                args.torch_dir,
                args.libtensorflow_dir,
                verbose=verbose,
            )
    except (SetupError, BuildError) as e:
        logger.error(str(e))
        return 1

    backends = installed_redisai_backends()
    backends_str = ", ".join(s.capitalize() for s in backends) if backends else "No"
    logger.info(f"{backends_str} backend(s) built")

    try:
        if "torch" in backends:
            check_py_torch_version(versions, device)
        if "tensorflow" in backends:
            check_py_tf_version(versions)
        if "onnxruntime" in backends:
            check_py_onnx_version(versions)
    except (SetupError, BuildError) as e:
        logger.error(str(e))
        return 1

    logger.info("SmartSim build complete!")
    return 0


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
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device to build ML runtimes for",
    )
    parser.add_argument(
        "--only_python_packages",
        action="store_true",
        default=False,
        help="Only evaluate the python packages (i.e. skip building backends)",
    )
    parser.add_argument(
        "--no_pt",
        action="store_true",
        default=False,
        help="Do not build PyTorch backend",
    )
    parser.add_argument(
        "--no_tf",
        action="store_true",
        default=False,
        help="Do not build TensorFlow backend",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        default=False,
        help="Build ONNX backend (off by default)",
    )
    parser.add_argument(
        "--torch_dir",
        default=None,
        type=str,
        help=f"Path to custom <path>/torch/share/cmake/Torch/ directory {warn_usage}",
    )
    parser.add_argument(
        "--libtensorflow_dir",
        default=None,
        type=str,
        help=f"Path to custom libtensorflow directory {warn_usage}",
    )
    parser.add_argument(
        "--keydb",
        action="store_true",
        default=False,
        help="Build KeyDB instead of Redis",
    )
