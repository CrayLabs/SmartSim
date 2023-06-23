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

import typing as t
from tabulate import tabulate

from smartsim._core._cli.utils import color_bool, pip_install, pip_uninstall
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
                # XXX: focrce install??
                if onnx:
                    self.install_onnx_wheels(force=True)
                if tf:
                    self.install_tf_wheel(force=True)
                if pt:
                    self.install_torch(device=args.device, force=True)
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

        # Check for onnx and tf in user python environemnt or download
        # them if requested. If they are not present and the user did not request
        # they be downloaded, we still should not break the build, as we use
        # onnx and tf directly from RAI instead of pip like we do PyTorch.
        if onnx:
            self.install_onnx_wheels(force=True)
        if tf:
            self.install_tf_wheel(force=True)

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
                self.install_torch(device=device, force=True)
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

    def install_torch(self, device: str = "cpu", force: bool = False) -> None:
        """Torch shared libraries installed by pip are used in the build
        for SmartSim backends so we download them here.
        """
        logger.info(f"Searching for a compatible TORCH install...")
        if self.build_env.is_macos():
            end_point = None
            device_suffix = ""
        else:  # linux
            end_point="https://download.pytorch.org/whl/torch_stable.html"
            if device == "cpu":
                device_suffix = self.versions.TORCH_CPU_SUFFIX
            elif device in ["gpu", "cuda"]:
                device_suffix = self.versions.TORCH_CUDA_SUFFIX
            else:
                raise BuildError("Unrecognized device requested")

        torch_packages = {
            "torch": f"{self.versions.TORCH}{device_suffix}",
            "torchvision": f"{self.versions.TORCHVISION}{device_suffix}",
            }

        def torch_validator(package: str, version: t.Optional[str]) -> bool:
            if not self.build_env.check_installed(package, version):
                return False
            # Previous check only looks at major/minor version numbers,
            # Torch requires we look at the patch as well
            installed = self.build_env.get_package_version(package)
            if device_suffix and device_suffix not in installed.patch:
                # XXX: Check w/ andrew, prev this was only enforced for cpu
                # Torch requires that we compare to the patch
                raise SetupError(f"{package}=={installed} does not satisfy device "
                                 f"suffix requirement: {device_suffix}")
            return True

        self._install_py_wheels(
            torch_packages,
            end_point=end_point,
            validator=torch_validator,
            force=force)

    def install_onnx_wheels(self, force: bool = False) -> None:
        """Check Python environment for a compatible ONNX installation"""
        logger.info(f"Searching for a compatible python ONNX install...")
        if not self.versions.ONNX:
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
        self._install_py_wheels({
            "onnx": f"{self.versions.ONNX}",
            "skl2onnx": f"{self.versions.REDISAI.skl2onnx}",
            "onnxmltools": f"{self.versions.REDISAI.onnxmltools}",
            "scikit-learn": f"{self.versions.REDISAI.__getattr__('scikit-learn')}",
        }, force=force)

    def install_tf_wheel(self, force: bool = False) -> None:
        """Check Python environment for a compatible TensorFlow installation"""
        logger.info(f"Searching for a compatible TF install...")
        self._install_py_wheels({"tensorflow": self.versions.TENSORFLOW},
                 force=force)

    def _install_py_wheels(self, 
            packages: t.Mapping[str, t.Optional[str]], 
            end_point: t.Optional[str] = None,
            validator: t.Optional[t.Callable[[str, t.Optional[str]], bool]] = None,
            force: bool = False) -> None:
        to_uninstall: t.List[str] = []
        to_install: t.List[str] = []
        validator = validator or self.build_env.check_installed

        for name, version in packages.items():
            spec = f"{name}=={version}" if version else name
            try:
                if validator(name, version):
                    # Installed at the correct version, nothing to do here
                    logger.info(f"{spec} already installed in Python environment")
                else:
                    # Not installed, install prefered version
                    to_install.append(spec)
            except SetupError as e:
                # Incompatible version found
                logger.warning(str(e))
                if force:
                    logger.info(f"Queueing {name} for reinstall")
                    to_uninstall.append(name)
                    to_install.append(spec)
        if to_uninstall:
            pip_uninstall(to_uninstall, verbose=self.verbose)
        if to_install:
            pip_install(to_install, end_point=end_point, verbose=self.verbose)

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
