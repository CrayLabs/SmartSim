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

from smartsim._core._cli.utils import (
    color_bool,
    pip_install,
    pip_uninstall,
    smart_logger_format,
)
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

logger = get_logger("Smart", fmt=smart_logger_format)

# NOTE: all smartsim modules need full paths as the smart cli
#       may be installed into a different directory.


_TDeviceStr = t.Literal["cpu", "gpu"]
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
        build_env(), build_env.MALLOC, build_env.JOBS, verbose
    )
    if not database_builder.is_built:
        logger.info(
            f"Building {database_name} version {versions.REDIS} from {versions.REDIS_URL}"
        )
        database_builder.build_from_git(versions.REDIS_URL, versions.REDIS_BRANCH)
        database_builder.cleanup()
    logger.info(f"{database_name} build complete!")


def build_redis_ai(
    build_env: BuildEnv,
    versions: Versioner,
    device: _TDeviceStr,
    torch: bool = True,
    tf: bool = True,
    onnx: bool = False,
    torch_dir: t.Union[str, Path, None] = None,
    libtf_dir: t.Union[str, Path, None] = None,
    modify_python_env: bool = False,
    verbose: bool = False,
) -> None:
    # make sure user isn't trying to do something silly on MacOS
    if build_env.PLATFORM == "darwin" and device == "gpu":
        raise BuildError("SmartSim does not support GPU on MacOS")

    # decide which runtimes to build
    print("\nML Backends Requested")
    backends_table = [
        ["PyTorch", versions.TORCH, color_bool(torch)],
        ["TensorFlow", versions.TENSORFLOW, color_bool(tf)],
        ["ONNX", versions.ONNX or "Unavailable", color_bool(onnx)],
    ]
    print(tabulate(backends_table, tablefmt="fancy_outline"), end="\n\n")
    print(f"Building for GPU support: {color_bool(device == 'gpu')}\n")

    if not check_backends_install():
        sys.exit(1)

    # Check for onnx and tf in user python environemnt and prompt user
    # to download them if they are not installed. this should not break
    # the build however, as we use onnx and tf directly from RAI instead
    # of pip like we do PyTorch.
    if onnx:
        install_py_onnx_version(fetch_if_missing=modify_python_env, verbose=verbose)
    if tf:
        install_py_tf_version(fetch_if_missing=modify_python_env, verbose=verbose)

    # TORCH
    if torch:
        if torch_dir:
            torch_dir = Path(torch_dir).resolve()
            if not torch_dir.is_dir():
                # we will always be able to find a torch version downloaded by
                # pip so if we can't find it we know the user suggested a torch
                # installation path that doesn't exist
                raise SetupError("Could not find requested user Torch installation")
        else:
            # install pytorch wheel, and get the path to the cmake dir
            # we will use in the RAI build
            install_py_torch_version(
                device=device,
                fetch_if_missing=modify_python_env,
                verbose=verbose,
            )
            torch_dir = build_env.torch_cmake_path

    if tf and libtf_dir:
        libtf_dir = Path(libtf_dir).resolve()

    build_env_dict = build_env()

    rai_builder = builder.RedisAIBuilder(
        build_env=build_env_dict,
        torch_dir=str(torch_dir) if torch_dir else "",
        libtf_dir=str(libtf_dir) if libtf_dir else "",
        build_torch=torch,
        build_tf=tf,
        build_onnx=onnx,
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


def infer_torch_device() -> _TDeviceStr:
    backend_torch_path = f"{CONFIG.lib_path}/backends/redisai_torch"
    return (
        "gpu" if Path(f"{backend_torch_path}/lib/libtorch_cuda.so").is_file() else "cpu"
    )


def install_py_torch_version(
    device: _TDeviceStr = "cpu",
    fetch_if_missing: bool = False,
    verbose: bool = False,
) -> None:
    """Torch shared libraries installed by pip are used in the build
    for SmartSim backends so we download them here.
    """
    logger.info(f"Searching for a compatible TORCH install...")
    if BuildEnv.is_macos():
        end_point = None
        device_suffix = ""
    else:  # linux
        end_point = "https://download.pytorch.org/whl/torch_stable.html"
        if device == "cpu":
            device_suffix = Versioner.TORCH_CPU_SUFFIX
        elif device == "gpu":
            device_suffix = Versioner.TORCH_CUDA_SUFFIX
        else:
            raise BuildError("Unrecognized device requested")

    torch_packages = {
        "torch": Version_(f"{Versioner.TORCH}{device_suffix}"),
        "torchvision": Version_(f"{Versioner.TORCHVISION}{device_suffix}"),
    }

    _confirm_package_in_python_env(
        torch_packages,
        end_point=end_point,
        validate_installed_version=_create_torch_version_validator(
            with_suffix=device_suffix
        ),
        install_on_absent=fetch_if_missing,
        install_on_conflict=False,
        verbose=verbose,
    )


def _create_torch_version_validator(
    with_suffix: str,
) -> t.Callable[[str, t.Optional[Version_]], bool]:
    def check_torch_version(package: str, version: t.Optional[Version_]) -> bool:
        if not BuildEnv.check_installed(package, version):
            return False
        # Default check only looks at major/minor version numbers,
        # Torch requires we look at the patch as well
        installed = BuildEnv.get_package_version(package)
        if with_suffix and with_suffix not in installed.patch:
            # Torch requires that we compare to the patch
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


def install_py_onnx_version(
    fetch_if_missing: bool = False,
    verbose: bool = False,
) -> None:
    """Check Python environment for a compatible ONNX installation"""
    logger.info("Searching for a compatible ONNX install...")
    if not Versioner.ONNX:
        py_version = sys.version_info
        msg = (
            "An onnx wheel is not available for "
            f"Python {py_version.major}.{py_version.minor}. "
            "Instead consider using Python 3.8 or 3.9 with RedisAI "
        )
        if sys.platform == "linux":
            msg += "1.2.5 or "
        msg += "1.2.7."
        raise SetupError(msg)
    _confirm_package_in_python_env(
        {
            "onnx": Version_(f"{Versioner.ONNX}"),
            "skl2onnx": Version_(f"{Versioner.REDISAI.skl2onnx}"),
            "onnxmltools": Version_(f"{Versioner.REDISAI.onnxmltools}"),
            "scikit-learn": Version_(f"{getattr(Versioner.REDISAI, 'scikit-learn')}"),
        },
        install_on_absent=fetch_if_missing,
        install_on_conflict=False,
        verbose=verbose,
    )


def install_py_tf_version(
    fetch_if_missing: bool = False,
    verbose: bool = False,
) -> None:
    """Check Python environment for a compatible TensorFlow installation"""
    logger.info(f"Searching for a compatible TF install...")
    _confirm_package_in_python_env(
        {"tensorflow": Versioner.TENSORFLOW},
        install_on_absent=fetch_if_missing,
        install_on_conflict=False,
        verbose=verbose,
    )


def _confirm_package_in_python_env(
    packages: t.Mapping[str, t.Optional[Version_]],
    package_pinning: _TPinningStr = "==",
    end_point: t.Optional[str] = None,
    validate_installed_version: t.Optional[
        t.Callable[[str, t.Optional[Version_]], bool]
    ] = None,
    install_on_absent: bool = False,
    install_on_conflict: bool = False,
    verbose: bool = False,
) -> None:
    # TODO: Do not like how the defualt validation function will always look for
    #       a `==` pinning. Maybe turn `BuildEnv.check_installed` into a factory
    #       that takes a pinning and an appropiate validation fn?
    validate_installed_version = validate_installed_version or BuildEnv.check_installed
    missing, conflicts, to_install, to_uninstall = _assess_python_env(
        packages,
        package_pinning,
        validate_installed_version,
        install_on_absent,
        install_on_conflict,
        verbose=verbose,
    )

    if missing or conflicts:
        indent = "\n\t"
        fmt_list: t.Callable = lambda n, l: f"{n}:{indent}{indent.join(l)}" if l else ""
        missing_str = fmt_list("Missing", missing)
        conflict_str = fmt_list("Conflicting", conflicts)
        sep = "\n" if missing_str and conflict_str else ""
        logger.warning(
            "Python Env Status Warning!\n"
            "Requested Packages are Missing or Conflicting:\n\n"
            f"{missing_str}{sep}{conflict_str}"
            "\n\nConsider installing packages at the requested versions via "
            "`pip` or re-running `smart build` with `--modify_python_env`"
        )
    if to_uninstall:
        pip_uninstall(to_uninstall, verbose=verbose)
    if to_install:
        pip_install(to_install, end_point=end_point, verbose=verbose)


def _assess_python_env(
    packages: t.Mapping[str, t.Optional[Version_]],
    package_pinning: _TPinningStr,
    validate_installed_version: t.Callable[[str, t.Optional[Version_]], bool],
    install_on_absent: bool,
    install_on_conflict: bool,
    verbose: bool,
) -> t.Tuple[t.List[str], t.List[str], t.List[str], t.List[str]]:
    missing: t.List[str] = []
    conflicts: t.List[str] = []
    to_uninstall: t.List[str] = []
    to_install: t.List[str] = []

    verbose_info: t.Callable[[str], t.Any] = lambda s: ...
    if verbose:
        verbose_info = logger.info

    for name, version in packages.items():
        spec = f"{name}{package_pinning}{version}" if version else name
        try:
            if validate_installed_version(name, version):
                # Installed at the correct version, nothing to do here
                verbose_info(f"{spec} already installed in environment")
            else:
                # Not installed! Install preferred version or warn user
                if install_on_absent:
                    verbose_info(f"Package not found: Queueing `{spec}` for install")
                    to_install.append(spec)
                else:
                    missing.append(spec)
        except VersionConflictError as e:
            # Incompatible version found
            if install_on_conflict:
                verbose_info(f"{e}: Queueing `{name}` for reinstall")
                to_uninstall.append(name)
                to_install.append(spec)
            else:
                conflicts.append(spec)

    return missing, conflicts, to_install, to_uninstall


def execute(args: argparse.Namespace) -> int:
    verbose = args.v
    keydb = args.keydb
    device: _TDeviceStr = args.device

    # torch and tf build by default
    pt = not args.no_pt
    tf = not args.no_tf
    onnx = args.onnx

    do_checks = not args.only_python_packages
    modify_python_env = args.modify_python_env
    build_env = BuildEnv(checks=do_checks)
    logger.info("Running SmartSim build process...")

    try:
        logger.info("Checking requested versions...")
        versions = Versioner()

        if args.only_python_packages:
            logger.info("Only installing Python packages...skipping build")
            if onnx:
                install_py_onnx_version(
                    fetch_if_missing=modify_python_env,
                    verbose=verbose,
                )
            if tf:
                install_py_tf_version(
                    fetch_if_missing=modify_python_env,
                    verbose=verbose,
                )
            if onnx:
                install_py_torch_version(
                    fetch_if_missing=modify_python_env,
                    verbose=verbose,
                )
        else:
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
                modify_python_env=modify_python_env,
                verbose=verbose,
            )

            backends = [
                backend.capitalize() for backend in installed_redisai_backends()
            ]
            logger.info(
                (", ".join(backends) if backends else "No") + " backend(s) built"
            )

    except (SetupError, BuildError) as e:
        logger.error(str(e))
        return 1

    logger.info("SmartSim build complete!")
    return 0


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Builds the parser for the command"""
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
        help="Path to custom <path>/torch/share/cmake/Torch/ directory (ONLY USE IF NEEDED)",
    )
    parser.add_argument(
        "--libtensorflow_dir",
        default=None,
        type=str,
        help="Path to custom libtensorflow directory (ONLY USED IF NEEDED)",
    )
    parser.add_argument(
        "--only_python_packages",
        action="store_true",
        default=False,
        help="If true, only install the python packages (i.e. skip backend builds)",
    )
    parser.add_argument(
        "--modify_python_env",
        action="store_true",
        default=False,
        help=(
            "If true, `smart` will use `pip` to attempt to modify the current "
            "python environment to satisfy the package dependencies of SmartSim "
            "and RedisAI"
        ),
    )
    parser.add_argument(
        "--keydb",
        action="store_true",
        default=False,
        help="Build KeyDB instead of Redis",
    )
