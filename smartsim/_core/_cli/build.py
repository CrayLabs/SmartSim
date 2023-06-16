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
from pathlib import Path

import pkg_resources
import typing as t
from tabulate import tabulate

from smartsim._core._cli.utils import color_bool, pip_install
from smartsim._core._install import builder
from smartsim._core._install.buildenv import BuildEnv, SetupError, Version_, Versioner, DbEngine
from smartsim._core._install.builder import BuildError
from smartsim._core.config import CONFIG
from smartsim._core.utils.helpers import installed_redisai_backends
from smartsim.error import SSConfigError
from smartsim.log import get_logger

smart_logger_format = "[%(name)s] %(levelname)s %(message)s"
logger = get_logger("Smart", fmt=smart_logger_format)

# NOTE: all smartsim modules need full paths as the smart cli
#       may be installed into a different directory.



def _install_torch_from_pip(versions: Versioner, device: str = "cpu", verbose: bool = False) -> None:

    packages = []
    end_point = None

    if sys.platform == "darwin":
        if device == "gpu":
            logger.warning("GPU support is not available on Mac OS X")
        # The following is deliberately left blank as there is no
        # alternative package available on Mac OS X
        device_suffix = ""
        end_point = None

    # if we are on linux cpu, either CUDA or CPU must be installed
    elif sys.platform == "linux":
        end_point = "https://download.pytorch.org/whl/torch_stable.html"
        if device in ["gpu", "cuda"]:
            device_suffix = versions.TORCH_CUDA_SUFFIX
        elif device == "cpu":
            device_suffix = versions.TORCH_CPU_SUFFIX

    packages.append(f"torch=={versions.TORCH}{device_suffix}")
    packages.append(f"torchvision=={versions.TORCHVISION}{device_suffix}")

    pip_install(packages, end_point=end_point, verbose=verbose)


class Build:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
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
            help="Device to build ML runtimes for (cpu || gpu)",
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
            "--keydb",
            action="store_true",
            default=False,
            help="Build KeyDB instead of Redis",
        )
        args = parser.parse_args(sys.argv[2:])
        self.verbose = args.v
        self.keydb = args.keydb

        # torch and tf build by default
        pt = not args.no_pt
        tf = not args.no_tf
        onnx = args.onnx

        logger.info("Running SmartSim build process...")
        try:
            logger.info("Checking requested versions...")
            self.versions = Versioner()

            if args.only_python_packages:
                logger.info("Only installing Python packages...skipping build")
                self.build_env = BuildEnv(checks=False)
                if not args.no_pt:
                    self.install_torch(device=args.device)
            else:
                logger.info("Checking for build tools...")
                self.build_env = BuildEnv()
                
                if self.verbose:
                    logger.info("Build Environment:")
                    env = self.build_env.as_dict()
                    env_vars = list(env.keys())
                    print(tabulate(env, headers=env_vars, tablefmt="github"), "\n")

                if self.keydb:
                    self.versions.REDIS = Version_("6.2.0")
                    self.versions.REDIS_URL = "https://github.com/EQ-Alpha/KeyDB"
                    self.versions.REDIS_BRANCH = "v6.2.0"
                    CONFIG.conf_path = Path(CONFIG.core_path, "config", "keydb.conf")
                    if not CONFIG.conf_path.resolve().is_file():
                        raise SSConfigError(
                            "Database configuration file at REDIS_CONF could not be found"
                        )

                if self.verbose:
                    db_name: DbEngine = "KEYDB" if self.keydb else "REDIS"
                    logger.info("Version Information:")
                    vers = self.versions.as_dict(db_name=db_name)
                    version_names = list(vers.keys())
                    print(tabulate(vers, headers=version_names, tablefmt="github"), "\n")

                # REDIS/KeyDB
                self.build_database()

                # REDISAI
                self.build_redis_ai(
                    str(args.device),
                    pt,
                    tf,
                    onnx,
                    args.torch_dir,
                    args.libtensorflow_dir,
                )

                backends = [
                    backend.capitalize() for backend in installed_redisai_backends()
                ]
                logger.info(
                    (", ".join(backends) if backends else "No") + " backend(s) built"
                )

        except (SetupError, BuildError) as e:
            logger.error(str(e))
            sys.exit(1)

        logger.info("SmartSim build complete!")

    def build_database(self) -> None:
        # check database installation
        database_name = "KeyDB" if self.keydb else "Redis"
        database_builder = builder.DatabaseBuilder(
            self.build_env(), self.build_env.MALLOC, self.build_env.JOBS, self.verbose
        )
        if not database_builder.is_built:
            logger.info(
                f"Building {database_name} version {self.versions.REDIS} from {self.versions.REDIS_URL}"
            )
            database_builder.build_from_git(
                self.versions.REDIS_URL, self.versions.REDIS_BRANCH
            )
            database_builder.cleanup()
        logger.info(f"{database_name} build complete!")

    def build_redis_ai(
        self,
        device: str,
        torch: bool = True,
        tf: bool = True,
        onnx: bool = False,
        torch_dir: t.Union[str, Path, None] = None,
        libtf_dir: t.Union[str, Path, None] = None
    ) -> None:

        # make sure user isn't trying to do something silly on MacOS
        if self.build_env.PLATFORM == "darwin" and device == "gpu":
            raise BuildError("SmartSim does not support GPU on MacOS")

        # decide which runtimes to build
        print("\nML Backends Requested")
        backends_table = [
            ["PyTorch", self.versions.TORCH, color_bool(torch)],
            ["TensorFlow", self.versions.TENSORFLOW, color_bool(tf)],
            ["ONNX", self.versions.ONNX or "Unavailable", color_bool(onnx)],
        ]
        print(tabulate(backends_table, tablefmt="fancy_outline"), end="\n\n")
        print(f"Building for GPU support: {color_bool(device == 'gpu')}\n")

        self.check_backends_install()

        # Check for onnx and tf in user python environemnt and prompt user
        # to download them if they are not installed. this should not break
        # the build however, as we use onnx and tf directly from RAI instead
        # of pip like we do PyTorch.
        if onnx:
            self.check_onnx_install()
        if tf:
            self.check_tf_install()

        # TORCH
        if torch:
            if torch_dir:
                torch_dir = Path(torch_dir).resolve()
                if not torch_dir.is_dir():
                    # we will always be able to find a torch version downloaded by
                    # pip so if we can't find it we know the user suggested a torch
                    # installation path that doesn't exist
                    logger.error("Could not find requested user Torch installation")
                    sys.exit(1)
            else:
                # install pytorch wheel, and get the path to the cmake dir
                # we will use in the RAI build
                self.install_torch(device=device)
                torch_dir = self.build_env.torch_cmake_path

        if tf:
            if libtf_dir:
                libtf_dir = Path(libtf_dir).resolve()

        rai_builder = builder.RedisAIBuilder(
            build_env=self.build_env(),
            torch_dir=str(torch_dir) if torch_dir else "",
            libtf_dir=str(libtf_dir) if libtf_dir else "",
            build_torch=torch,
            build_tf=tf,
            build_onnx=onnx,
            jobs=self.build_env.JOBS,
            verbose=self.verbose,
        )

        if rai_builder.is_built:
            logger.info("RedisAI installed. Run `smart clean` to remove.")
        else:
            # get the build environment, update with CUDNN env vars
            # if present and building for GPU, otherwise warn the user
            build_env = self.build_env()
            if device == "gpu":
                gpu_env = self.build_env.get_cudnn_env()
                cudnn_env_vars = [
                    "CUDNN_LIBRARY",
                    "CUDNN_INCLUDE_DIR",
                    "CUDNN_INCLUDE_PATH",
                    "CUDNN_LIBRARY_PATH",
                ]
                if not gpu_env:
                    logger.warning(
                        f"CUDNN environment variables not found.\n"
                        + f"Looked for {cudnn_env_vars}"
                    )
                else:
                    build_env.update(gpu_env)
            # update RAI build env with cudnn env vars
            rai_builder.env = build_env

            logger.info(
                f"Building RedisAI version {self.versions.REDISAI}"
                f" from {self.versions.REDISAI_URL}"
            )

            # NOTE: have the option to add other builds here in the future
            # like "from_tarball"
            rai_builder.build_from_git(
                self.versions.REDISAI_URL, self.versions.REDISAI_BRANCH, device
            )
            logger.info("ML Backends and RedisAI build complete!")

    def infer_torch_device(self) -> str:
        backend_torch_path = f"{CONFIG.lib_path}/backends/redisai_torch"
        device = "cpu"
        if Path(f"{backend_torch_path}/lib/libtorch_cuda.so").is_file():
            device = "gpu"
        return device

    def install_torch(self, device: str = "cpu") -> None:
        """Torch shared libraries installed by pip are used in the build
        for SmartSim backends so we download them here.
        """

        if not self.build_env.check_installed("torch", self.versions.TORCH):
            inferred_device = self.infer_torch_device()
            if (inferred_device == "gpu") and (device == "cpu"):
                logger.warning("CPU requested, but GPU backend is available")
            _install_torch_from_pip(self.versions, device, self.verbose)
        # if torch already installed, check the versions to make sure correct
        # torch version is downloaded for that particular device
        else:
            installed = Version_(pkg_resources.get_distribution("torch").version)
            if device == "gpu":
                # if torch version is x.x.x+cpu
                if "cpu" in installed.patch:
                    msg = (
                        "Torch CPU is currently installed but torch GPU requested. Uninstall all torch packages "
                        + "and run the `smart build` command again to obtain Torch GPU libraries"
                    )
                    logger.warning(msg)

            if device == "cpu":
                # if torch version if x.x.x then we need to install the cpu version
                if "cpu" not in installed.patch and not self.build_env.is_macos():
                    msg = (
                        "Torch GPU installed in python environment but requested Torch CPU. "
                        + " Run `pip uninstall torch torchvision` and run `smart build` again"
                    )
                    logger.error(msg)  # error because this is usually fatal
            logger.info(f"Torch {self.versions.TORCH} installed in Python environment")

    def check_onnx_install(self) -> None:
        """Check Python environment for ONNX installation"""
        if not self.versions.ONNX:
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
        try:
            if not self.build_env.check_installed("onnx", self.versions.ONNX):
                msg = (
                    f"ONNX {self.versions.ONNX} not installed in python environment. "
                    + f"Consider installing onnx=={self.versions.ONNX} with pip"
                )
                logger.warning(msg)
            else:
                logger.info(
                    f"ONNX {self.versions.ONNX} installed in Python environment"
                )
        except SetupError as e:
            logger.warning(str(e))

    def check_tf_install(self) -> None:
        """Check Python environment for TensorFlow installation"""

        try:
            if not self.build_env.check_installed(
                "tensorflow", self.versions.TENSORFLOW
            ):
                msg = (
                    f"TensorFlow {self.versions.TENSORFLOW} not installed in Python environment. "
                    + f"Consider installing tensorflow=={self.versions.TENSORFLOW} with pip"
                )
                logger.warning(msg)
            else:
                logger.info(
                    f"TensorFlow {self.versions.TENSORFLOW} installed in Python environment"
                )
        except SetupError as e:
            logger.warning(str(e))

    def check_backends_install(self) -> None:
        """Checks if backends have already been installed.
        Logs details on how to proceed forward
        if the RAI_PATH environment variable is set or if
        backends have already been installed.
        """
        if os.environ.get("RAI_PATH"):
            if installed_redisai_backends():
                logger.error(
                    f"There is no need to build. Backends are already built and specified in the environment at 'RAI_PATH': {CONFIG.redisai}"
                )
            else:
                logger.error(
                    f"Before running 'smart build', unset your RAI_PATH environment variable with 'unset RAI_PATH'."
                )
            sys.exit(1)
        else:
            if installed_redisai_backends():
                logger.error(
                    "If you wish to re-run `smart build`, you must first run `smart clean`."
                )
                sys.exit(1)
