import sys
import argparse
from pathlib import Path

import pkg_resources

from smartsim._core._cli.utils import SetupError, color_bool, pip_install
from smartsim._core._install import builder
from smartsim._core._install.buildenv import BuildEnv, Versioner
from smartsim.log import get_logger

smart_logger_format = "[%(name)s] %(levelname)s %(message)s"
logger = get_logger("Smart", fmt=smart_logger_format)

# NOTE: all smartsim modules need full paths as the smart cli
#       may be installed into a different directory.


class Build:
    def __init__(self):
        self.build_env = BuildEnv()
        self.versions = Versioner()

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-v",
            action="store_true",
            default=False,
            help="Enable verbose build process",
        )
        parser.add_argument(
            "--device",
            type=str,
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
        args = parser.parse_args(sys.argv[2:])
        self.verbose = args.v

        # torch and tf build by default
        pt = not args.no_pt
        tf = not args.no_tf
        onnx = args.onnx

        logger.info("Running SmartSim build process...")
        logger.info("Build Environment:")
        print(str(self.build_env), "\n")

        # REDIS
        self.build_redis()

        # REDISAI
        self.build_redis_ai(str(args.device), pt, tf, onnx, args.torch_dir)

        logger.info("SmartSim build complete!")

    def build_redis(self):
        # check redis installation
        redis_builder = builder.RedisBuilder(
            self.build_env(),
            self.build_env.MALLOC,
            self.build_env.JOBS,
            self.verbose
        )

        if not redis_builder.is_built:
            logger.info(
                f"Building Redis version {self.versions.REDIS} from {self.versions.REDIS_URL}"
            )

            redis_builder.build_from_git(self.versions.REDIS_URL, self.versions.REDIS)
            redis_builder.cleanup()
        logger.info("Redis build complete!")

    def build_redis_ai(self, device, torch=True, tf=True, onnx=False, torch_dir=None):

        # decide which runtimes to build
        print("\nML Backends Requested")
        print("-----------------------")
        print(f"    PyTorch {self.versions.TORCH}: {color_bool(torch)}")
        print(f"    TensorFlow {self.versions.TENSORFLOW}: {color_bool(tf)}")
        print(f"    ONNX {self.versions.ONNX}: {color_bool(onnx)}\n")
        print(f"Building for GPU support: {color_bool(device == 'gpu')}\n")

        # sanity check for platform and RAI deps
        self.build_env.check_prereq("git-lfs")

        # check to make sure user didn't request GPU build on Mac
        if self.build_env.PLATFORM == "darwin" and device == "gpu":
            raise SetupError("SmartSim does not support GPU on MacOS")

        # ONNX
        if onnx:
            self.check_onnx_install()

        # TF
        if tf == 1:
            self.check_tf_install()

        cmd = []
        # TORCH
        if torch:
            if torch_dir:
                torch_dir = Path(torch_dir).resolve()
                if not torch_dir.is_dir():
                    raise SetupError("Could not find requested user Torch installation")
            else:
                self.install_torch(device=device)
                torch_dir = self.build_env.torch_cmake_path

        rai_builder = builder.RedisAIBuilder(
            build_env=self.build_env(),
            torch_dir=str(torch_dir),
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
                cudnn_env_vars = ["CUDNN_LIBRARY",
                                  "CUDNN_INCLUDE_DIR",
                                  "CUDNN_INCLUDE_PATH",
                                  "CUDNN_LIBRARY_PATH"]
                if not gpu_env:
                    logger.warning(f"CUDNN environment variables not found.\n" \
                        f"Looked for {cudnn_env_vars}")
                else:
                    build_env.update(gpu_env)
            # update RAI build env
            rai_builder.env = build_env

            logger.info(
                f"Building RedisAI version {self.versions.REDISAI}" \
                    f" from {self.versions.REDISAI_URL}")

            rai_builder.build_from_git(
                self.versions.REDISAI_URL,
                self.versions.REDISAI,
                device
            )
            logger.info("ML Backends and RedisAI build complete!")

    def install_torch(self, device="cpu"):
        """Torch shared libraries installed by pip are used in the build
        for SmartSim backends so we download them here.

        :raises SetupError: if incompatible version is installed
        """
        packages = []
        end_point = None
        if not self.build_env.check_installed("torch", self.versions.TORCH):
            # if we are on linux cpu, use the torch without CUDA
            if sys.platform == "linux" and device == "cpu":
                packages.append(f"torch=={self.versions.TORCH}+cpu")
                packages.append(f"torchvision=={self.versions.TORCHVISION}+cpu")
                end_point = "https://download.pytorch.org/whl/torch_stable.html"

            # otherwise just use the version downloaded by pip
            else:
                packages.append(f"torch=={self.versions.TORCH}")
                packages.append(f"torchvision=={self.versions.TORCHVISION}")

            pip_install(packages, end_point=end_point, verbose=self.verbose)

        # if torch already installed, check the versions
        else:
            installed_ver = pkg_resources.get_distribution("torch").version
            _, _, patch = installed_ver.split(".")
            if "cpu" in patch and device == "gpu":
                msg = "Torch CPU is currently installed but torch GPU requested. Uninstall all torch packages"
                msg += (
                    " and run the `smart` command again to obtain Torch GPU libraries"
                )
                logger.warning(msg)
            if device == "cpu" and "cpu" not in patch and not self.build_env.is_macos():
                msg = (
                    "Torch GPU installed in python environment but requested Torch CPU."
                )
                msg += " Run `pip uninstall torch torchvision` and run `smart` again"
                logger.warning(msg)
            logger.info(f"Torch {self.versions.TORCH} installed in Python environment")

    def check_onnx_install(self):
        """Check Python environment for ONNX installation"""
        # conversions tools for ONNX
        packages = [
            f"skl2onnx=={self.versions.SKL2ONNX}",
            f"onnxmltools=={self.versions.ONNXML}",
            f"onnx=={self.versions.ONNX}",
        ]
        try:
            if not self.build_env.check_installed("onnx", self.versions.ONNX):
                msg = f"ONNX {self.versions.ONNX} not installed in python environment\n"
                msg += f"Consider installing {' '.join(packages)} with pip"
                logger.warning(msg)
            else:
                logger.info(f"ONNX {self.versions.ONNX} installed in Python environment")
        except SetupError as e:
            logger.warning(str(e))

    def check_tf_install(self):
        """Check Python environment for TensorFlow installation"""

        try:
            if not self.build_env.check_installed(
                "tensorflow", self.versions.TENSORFLOW
            ):
                msg = f"TensorFlow {self.versions.TENSORFLOW} not installed in Python environment\n"
                msg += f"Consider installing tensorflow=={self.versions.TENSORFLOW} with pip"
                logger.warning(msg)
            else:
                logger.info(
                    f"TensorFlow {self.versions.TENSORFLOW} installed in Python environment")
        except SetupError as e:
            logger.warning(str(e))