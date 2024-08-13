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

from collections import deque, namedtuple

import fileinput
import pathlib
import re
import shutil
import typing as t

from packaging.version import Version

from smartsim._core._cli.utils import SMART_LOGGER_FORMAT
from smartsim._core.config import CONFIG
from smartsim._core._install.buildenv import BuildEnv
from smartsim._core._install.platform import Platform
from smartsim._core._install.mlpackages import MLPackage
from smartsim._core._install.utils import PackageRetriever
from smartsim._core.utils.shell import execute_cmd
from smartsim.log import get_logger

logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)

class RedisAIBuilder():
    """Class to build RedisAI from Source
    Supported build method:
     - from git
    See buildenv.py for buildtime configuration of RedisAI
    version and url.
    """

    def __init__(
        self,
        platform: Platform,
        mlpackages: t.Dict[str, MLPackage],
        build_env: BuildEnv,
        verbose: bool = False,
        source: t.Union[str, pathlib.Path] = "https://github.com/RedisAI/RedisAI.git",
        version: str = "v1.2.7"
    ) -> None:
        self.platform = platform
        self.mlpackages = mlpackages
        self.build_env = build_env
        self.verbose = verbose
        self.source = source
        self.version = version
        self.patches: t.List[_RedisAIPatch] = []
        self._define_patches_by_version()

        self.src_path = CONFIG.build_path / "RedisAI" /"src"
        self.build_path = CONFIG.build_path / "RedisAI" / "build"
        self.package_path = CONFIG.build_path / "RedisAI" / "mlpackages"

        self.cleanup_build()

    def _define_patches_by_version(self):
        if self.build_torch:
            if Version(self.mlpackages["libtorch"].version) >= Version("2.1.0"):
                self.patches.append(_patches["c++17"])


    def cleanup_build(self):
        shutil.rmtree(self.src_path, ignore_errors=True)
        shutil.rmtree(self.build_path, ignore_errors=True)
        shutil.rmtree(self.package_path, ignore_errors=True)

    @property
    def is_built(self) -> bool:
        backend_dir = CONFIG.lib_path / "backends"
        rai_exists = [
            (backend_dir / f"redisai_{backend_name}").is_dir() for backend_name in self.mlpackages
        ]
        rai_exists.append((CONFIG.lib_path / "redisai.so").is_file())
        return all(rai_exists)

    @property
    def build_torch(self) -> bool:
        return "libtorch" in self.mlpackages

    @property
    def build_tensorflow(self) -> bool:
        return "libtensorflow" in self.mlpackages

    @property
    def build_onnxruntime(self) -> bool:
        return "onnxruntime" in self.mlpackages

    def build(self) -> None:
        """Build RedisAI

        :param git_url: url from which to retrieve RedisAI
        :param branch: branch to checkout
        :param device: cpu or gpu
        """

        self.src_path.mkdir()
        self.build_path.mkdir()
        self.package_path.mkdir()

        # Create the build directory structure
        git_kwargs = {
            "depth":1,
            "branch":self.version,
        }

        PackageRetriever.retrieve(self.source, self.src_path, **git_kwargs)
        self._patch_source_files()
        self._prepare_packages()

        cmake_command = self._rai_cmake_cmd()
        build_command = self._rai_build_cmd()

        logger.info(f"Configuring CMake Build: {cmake_command.join(' ')}")
        self.run_command(cmake_command, self.build_path)
        logger.info(f"Building RedisAI: {build_command.join(' ')}")
        self.run_command(build_command, self.build_path)

    def _prepare_packages(self):
        def find_closest_object(start_path, target_obj):
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
            if actual_root != target_dir:
                logger.info(f"Non-standard location found: {str(actual_root)} -> {str(target_dir)}")
                for f in actual_root.iterdir():
                    f.rename(target_dir / f.name)

    def run_command(self, cmd, cwd):
        rc, output, err, = execute_cmd(cmd, cwd=str(cwd))
        if (rc != 0) or self.verbose:
            print(output)
            print(err)
        if (rc != 0):
            raise Exception(f"Build Failed with code: {rc}")

    def _rai_cmake_cmd(self) -> t.List[str]:
        def on_off(expression: bool):
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

    def _rai_build_cmd(self):
        return "make install -j VERBOSE=1".split(" ")

    def _patch_source_files(self) -> None:
        for patch in self.patches:
            compiled_regex = re.compile(patch.regex)
            with fileinput.input(str(self.src_path/patch.source_file), inplace=True) as f:
                for line in f:
                    line = compiled_regex.sub(patch.replacement, line)
                    print(line, end="")

_RedisAIPatch = namedtuple("_RedisAIPatch", "source_file regex replacement")
_patches = {
    "c++17":_RedisAIPatch(
        "src/backends/libtorch_c/CMakeLists.txt",
        r"set_property\(TARGET\storch_c\sPROPERTY\sCXX_STANDARD\s(98|11|14)\)",
        "set_property(TARGET torch_c PROPERTY CXX_STANDARD 17)",
    ),
}