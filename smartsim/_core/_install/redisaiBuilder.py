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
import pathlib
import re
import shutil
import subprocess
import typing as t
from collections import deque, namedtuple

from packaging.version import Version

from smartsim._core._cli.utils import SMART_LOGGER_FORMAT
from smartsim._core._install.buildenv import BuildEnv
from smartsim._core._install.mlpackages import MLPackageCollection
from smartsim._core._install.platform import Platform
from smartsim._core._install.utils import PackageRetriever
from smartsim._core.config import CONFIG
from smartsim.log import get_logger

logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)


class RedisAIBuilder:
    """Class to build RedisAI from Source"""

    def __init__(
        self,
        platform: Platform,
        mlpackages: MLPackageCollection,
        build_env: BuildEnv,
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
        self.patches: t.List[_RedisAIPatch] = []
        self._define_patches_by_version()

        self.src_path = CONFIG.build_path / "RedisAI" / "src"
        self.build_path = CONFIG.build_path / "RedisAI" / "build"
        self.package_path = CONFIG.build_path / "RedisAI" / "mlpackages"

        self.cleanup_build()

    def _define_patches_by_version(self) -> None:
        """Inject specific patches due to package version numbers"""
        if self.build_torch:
            if Version(self.mlpackages["libtorch"].version) >= Version("2.1.0"):
                self.patches.append(_patches["c++17"])

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

        self.src_path.mkdir(parents=True)
        self.build_path.mkdir(parents=True)
        self.package_path.mkdir(parents=True)

        # Create the build directory structure
        git_kwargs = {
            "depth": 1,
            "branch": self.version,
        }

        PackageRetriever.retrieve(self.source, self.src_path, **git_kwargs)
        self._patch_source_files()
        self._prepare_packages()

        cmake_command = self._rai_cmake_cmd()
        build_command = self._rai_build_cmd()

        logger.info(f"Configuring CMake Build:")
        if self.verbose:
            print(" ".join(cmake_command))
        self.run_command(cmake_command, self.build_path)

        logger.info(f"Building RedisAI:")
        if self.verbose:
            print(" ".join(cmake_command))
        self.run_command(build_command, self.build_path)

    def _prepare_packages(self) -> None:
        """Ensure that retrieved archives/packages are in the expected location

        RedisAI requires that the root directory of the backend is at
        DEP_PATH/example_backend. Due to difficulties in retrieval methods and
        naming conventions from different sources, this cannot be standardized.
        Instead we try to find the parent of the "include" directory and assume
        this is the root.
        """

        def find_closest_object(
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

        for package in self.mlpackages.values():
            logger.info(f"Retrieving package: {package.name} {package.version}")
            target_dir = self.package_path / package.name
            package.retrieve(target_dir)
            # Move actual contents to root of the expected location
            actual_root = find_closest_object(target_dir, "include")
            if actual_root and actual_root != target_dir:
                logger.info(
                    f"Non-standard location found: {str(actual_root)} -> {str(target_dir)}"
                )
                for f in actual_root.iterdir():
                    f.rename(target_dir / f.name)

    def run_command(self, cmd: t.Union[str, t.List[str]], cwd: pathlib.Path) -> None:
        """Executor of commands usedi in the build

        :param cmd: The actual command to execute
        :param cwd: The working directory to execute in
        """
        stdout = None if self.verbose else subprocess.DEVNULL
        stderr = None if self.verbose else subprocess.PIPE
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=stdout, stderr=stderr)
        if proc.returncode != 0:
            print(proc.stderr.decode("utf-8"))

    def _rai_cmake_cmd(self) -> t.List[str]:
        """Build the CMake configuration command

        :return: CMake command with correct options
        """

        def on_off(expression: bool) -> t.Literal["ON", "OFF"]:
            return "ON" if expression else "OFF"

        cmake_args = dict(
            BUILD_TF=on_off(self.build_tensorflow),
            BUILD_ORT=on_off(self.build_onnxruntime),
            BUILD_TORCH=on_off(self.build_torch),
            BUILD_TFLITE="OFF",
            DEPS_PATH=str(self.package_path),
            DEVICE="gpu" if self.platform.device.is_gpu() else "cpu",
            INSTALL_PATH=str(CONFIG.lib_path),
            CMAKE_C_COMPILER=self.build_env.CC,
            CMAKE_CXX_COMPILER=self.build_env.CXX,
        )
        cmd = ["cmake"]
        cmd += (f"-D{key}={value}" for key, value in cmake_args.items())
        cmd.append(str(self.src_path))
        return cmd

    def _rai_build_cmd(self) -> t.List[str]:
        """Shell command to build RedisAI and modules

        With the CMake based install, very little needs to be done here.
        "make install" is used to ensure that all resulting RedisAI backends
        and their dependencies end up in the same location with the correct
        RPATH if applicable.

        :return: Command used to compile RedisAI and backends
        """
        return "make install -j VERBOSE=1".split(" ")

    def _patch_source_files(self) -> None:
        """Apply all _RedisAIPatches specified previously"""
        for patch in self.patches:
            compiled_regex = re.compile(patch.regex)
            with fileinput.input(
                str(self.src_path / patch.source_file), inplace=True
            ) as f:
                for line in f:
                    line = compiled_regex.sub(patch.replacement, line)
                    print(line, end="")


_RedisAIPatch = namedtuple("_RedisAIPatch", "source_file regex replacement")
_patches = {
    "c++17": _RedisAIPatch(
        "src/backends/libtorch_c/CMakeLists.txt",
        r"set_property\(TARGET\storch_c\sPROPERTY\sCXX_STANDARD\s(98|11|14)\)",
        "set_property(TARGET torch_c PROPERTY CXX_STANDARD 17)",
    ),
}
