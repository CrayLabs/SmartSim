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

import pathlib
import shutil
import typing as t

from smartsim._core.config import CONFIG
from smartsim._core._install.platform import Platform
from smartsim._core._install.mlpackages import MLPackage
from smartsim._core._install.utils import PackageRetriever
from smartsim._core.utils.shell import execute_cmd
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
        build_env: t.Optional[t.Dict[str, str]] = None,
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

        self.src_path = CONFIG.build_path / "RedisAI"
        self.build_path = self.src_path / "build"
        self.package_path = self.src_path / "mlpackages"

    def cleanup(self):
        shutil.rmtree(self.src_path)

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
        return "torch" in self.mlpackages

    @property
    def build_tensorflow(self) -> bool:
        return "tensorflow" in self.mlpackages

    @property
    def build_onnxruntime(self) -> bool:
        return "onnx" in self.mlpackages

    def build(self) -> None:
        """Build RedisAI

        :param git_url: url from which to retrieve RedisAI
        :param branch: branch to checkout
        :param device: cpu or gpu
        """

        # Create the build directory structure
        self.build_dir.parent.mkdir(parents=True)
        git_kwargs = {
            "depth":1,
            "branch":self.version,
        }
        PackageRetriever(self.source, self.build_dir, **git_kwargs)

        for package in self.mlpackages:
            package.retrieve(self.rai_package_path)

        cmake_command = self._rai_cmake_cmd()
        build_command = self._rai_build_cmd

        self.run_command(cmake_command, self.rai_build_path)
        self.run_command(build_command, self.rai_build_path)

    def run_command(self, cmd, cwd):
        rc, output, err, = execute_cmd(cmd, cwd=cwd, env=self.build_env)
        if (rc != 0) or self.verbose:
            print(output)
            print(err)

    def _rai_cmake_cmd(self) -> t.List[str]:
        def on_off(expression: bool):
            return "ON" if expression else "OFF"
        cmake_args = dict(
            BUILD_TF=on_off(self.build_tensorflow),
            BUILD_ORT=on_off(self.build_onnxruntime),
            BUILD_TORCH=on_off(self.build_torch),
            DEPS_PATH=self.rai_package_path,
            DEVICE="gpu" if self.platform.device.is_gpu() else "cpu",
            CMAKE_INSTALL_PREFIX=CONFIG.dependency_path
        )
        cmd = ["cmake"]
        cmd.append(f"-D{key}={value}" for key, value in cmake_args.items())
        cmd.append(self.src_path)
        return cmd
