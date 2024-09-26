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

import fileinput
import os
import pathlib
import shutil
import stat
import subprocess
import typing as t
from collections import deque

from smartsim._core._cli.utils import SMART_LOGGER_FORMAT
from smartsim._core._install.buildenv import BuildEnv
from smartsim._core._install.mlpackages import MLPackageCollection, RAIPatch
from smartsim._core._install.platform import OperatingSystem, Platform
from smartsim._core._install.utils import retrieve
from smartsim._core.config import CONFIG
from smartsim.log import get_logger

logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)
_SUPPORTED_ROCM_ARCH = "gfx90a"


class RedisAIBuildError(Exception):
    pass


class RedisAIBuilder:
    """Class to build RedisAI from Source"""

    def __init__(
        self,
        platform: Platform,
        mlpackages: MLPackageCollection,
        build_env: BuildEnv,
        main_build_path: pathlib.Path,
        verbose: bool = False,
        source: t.Union[str, pathlib.Path] = "https://github.com/RedisAI/RedisAI.git",
        version: str = "v1.2.7",
    ) -> None:

        self.platform = platform
        self.mlpackages = mlpackages
        self.build_env = build_env
        self.verbose = verbose
        self.source = source
        self.version = version
        self._root_path = main_build_path / "RedisAI"

        self.cleanup_build()

    @property
    def src_path(self) -> pathlib.Path:
        return pathlib.Path(self._root_path / "src")

    @property
    def build_path(self) -> pathlib.Path:
        return pathlib.Path(self._root_path / "build")

    @property
    def package_path(self) -> pathlib.Path:
        return pathlib.Path(self._root_path / "package")

    def cleanup_build(self) -> None:
        """Removes all directories associated with the build"""
        shutil.rmtree(self.src_path, ignore_errors=True)
        shutil.rmtree(self.build_path, ignore_errors=True)
        shutil.rmtree(self.package_path, ignore_errors=True)

    @property
    def is_built(self) -> bool:
        """Determine whether RedisAI and backends were built

        :return: True if all backends and RedisAI module are in
                 the expected location
        """
        backend_dir = CONFIG.lib_path / "backends"
        rai_exists = [
            (backend_dir / f"redisai_{backend_name}").is_dir()
            for backend_name in self.mlpackages
        ]
        rai_exists.append((CONFIG.lib_path / "redisai.so").is_file())
        return all(rai_exists)

    @property
    def build_torch(self) -> bool:
        """Whether to build torch backend

        :return: True if torch backend should be built
        """
        return "libtorch" in self.mlpackages

    @property
    def build_tensorflow(self) -> bool:
        """Whether to build tensorflow backend

        :return: True if tensorflow backend should be built
        """
        return "libtensorflow" in self.mlpackages

    @property
    def build_onnxruntime(self) -> bool:
        """Whether to build onnx backend

        :return: True if onnx backend should be built
        """
        return "onnxruntime" in self.mlpackages

    def build(self) -> None:
        """Build RedisAI

        :param git_url: url from which to retrieve RedisAI
        :param branch: branch to checkout
        :param device: cpu or gpu
        """

        # Following is needed to make sure that the clone/checkout is not
        # impeded by git LFS limits imposed by RedisAI
        os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"

        self.src_path.mkdir(parents=True)
        self.build_path.mkdir(parents=True)
        self.package_path.mkdir(parents=True)

        retrieve(self.source, self.src_path, depth=1, branch=self.version)

        self._prepare_packages()

        for package in self.mlpackages.values():
            self._patch_source_files(package.rai_patches)
        cmake_command = self._rai_cmake_cmd()
        build_command = self._rai_build_cmd

        if self.platform.device.is_rocm() and "libtorch" in self.mlpackages:
            pytorch_rocm_arch = os.environ.get("PYTORCH_ROCM_ARCH")
            if not pytorch_rocm_arch:
                logger.info(
                    f"PYTORCH_ROCM_ARCH not set. Defaulting to '{_SUPPORTED_ROCM_ARCH}'"
                )
                os.environ["PYTORCH_ROCM_ARCH"] = _SUPPORTED_ROCM_ARCH
            elif pytorch_rocm_arch != _SUPPORTED_ROCM_ARCH:
                logger.warning(
                    f"PYTORCH_ROCM_ARCH is not {_SUPPORTED_ROCM_ARCH} which is the "
                    "only officially supported architecture. This may still work "
                    "if you are supplying your own version of libtensorflow."
                )

        logger.info("Configuring CMake Build")
        if self.verbose:
            print(" ".join(cmake_command))
        self.run_command(cmake_command, self.build_path)

        logger.info("Building RedisAI")
        if self.verbose:
            print(" ".join(build_command))
        self.run_command(build_command, self.build_path)

        if self.platform.operating_system == OperatingSystem.LINUX:
            self._set_execute(CONFIG.lib_path / "redisai.so")

    @staticmethod
    def _set_execute(target: pathlib.Path) -> None:
        """Set execute permissions for file

        :param target: The target file to add execute permission
        """
        permissions = os.stat(target).st_mode | stat.S_IXUSR
        os.chmod(target, permissions)

    @staticmethod
    def _find_closest_object(
        start_path: pathlib.Path, target_obj: str
    ) -> t.Optional[pathlib.Path]:
        queue = deque([start_path])
        while queue:
            current_dir = queue.popleft()
            current_target = current_dir / target_obj
            if current_target.exists():
                return current_target.parent
            for sub_dir in current_dir.iterdir():
                if sub_dir.is_dir():
                    queue.append(sub_dir)
        return None

    def _prepare_packages(self) -> None:
        """Ensure that retrieved archives/packages are in the expected location

        RedisAI requires that the root directory of the backend is at
        DEP_PATH/example_backend. Due to difficulties in retrieval methods and
        naming conventions from different sources, this cannot be standardized.
        Instead we try to find the parent of the "include" directory and assume
        this is the root.
        """

        for package in self.mlpackages.values():
            logger.info(f"Retrieving package: {package.name} {package.version}")
            target_dir = self.package_path / package.name
            package.retrieve(target_dir)
            # Move actual contents to root of the expected location
            actual_root = self._find_closest_object(target_dir, "include")
            if actual_root and actual_root != target_dir:
                logger.debug(
                    (
                        "Non-standard location found: \n",
                        f"{actual_root} -> {target_dir}",
                    )
                )
                for file in actual_root.iterdir():
                    file.rename(target_dir / file.name)

    def run_command(self, cmd: t.Union[str, t.List[str]], cwd: pathlib.Path) -> None:
        """Executor of commands usedi in the build

        :param cmd: The actual command to execute
        :param cwd: The working directory to execute in
        """
        stdout = None if self.verbose else subprocess.DEVNULL
        stderr = None if self.verbose else subprocess.PIPE
        proc = subprocess.run(
            cmd, cwd=str(cwd), stdout=stdout, stderr=stderr, check=False
        )
        if proc.returncode != 0:
            if stderr:
                print(proc.stderr.decode("utf-8"))
            raise RedisAIBuildError(
                f"RedisAI build failed during command: {' '.join(cmd)}"
            )

    def _rai_cmake_cmd(self) -> t.List[str]:
        """Build the CMake configuration command

        :return: CMake command with correct options
        """

        def on_off(expression: bool) -> t.Literal["ON", "OFF"]:
            return "ON" if expression else "OFF"

        cmake_args = {
            "BUILD_TF": on_off(self.build_tensorflow),
            "BUILD_ORT": on_off(self.build_onnxruntime),
            "BUILD_TORCH": on_off(self.build_torch),
            "BUILD_TFLITE": "OFF",
            "DEPS_PATH": str(self.package_path),
            "DEVICE": "gpu" if self.platform.device.is_gpu() else "cpu",
            "INSTALL_PATH": str(CONFIG.lib_path),
            "CMAKE_C_COMPILER": self.build_env.CC,
            "CMAKE_CXX_COMPILER": self.build_env.CXX,
        }
        if self.platform.device.is_rocm():
            cmake_args["Torch_DIR"] = str(self.package_path / "libtorch")
        cmd = ["cmake"]
        cmd += (f"-D{key}={value}" for key, value in cmake_args.items())
        cmd.append(str(self.src_path))
        return cmd

    @property
    def _rai_build_cmd(self) -> t.List[str]:
        """Shell command to build RedisAI and modules

        With the CMake based install, very little needs to be done here.
        "make install" is used to ensure that all resulting RedisAI backends
        and their dependencies end up in the same location with the correct
        RPATH if applicable.

        :return: Command used to compile RedisAI and backends
        """
        return "make install -j VERBOSE=1".split(" ")

    def _patch_source_files(self, patches: t.Tuple[RAIPatch, ...]) -> None:
        """Apply specified RedisAI patches"""
        for patch in patches:
            with fileinput.input(
                str(self.src_path / patch.source_file), inplace=True
            ) as file_handle:
                for line in file_handle:
                    line = patch.regex.sub(patch.replacement, line)
                    print(line, end="")
