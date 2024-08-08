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

import importlib.resources as resources
import json
import pathlib
import subprocess
import sys
import typing as t

from pydantic import HttpUrl, BaseModel

from .types import PathLike
from .utils import PackageRetriever
from .platform import Platform

_PlatformPackages = t.TypeVar("_PlatformPackages", bound="PlatformPackages")

class MLPackage(BaseModel):
    name: str
    version: str
    pip_index: str
    packages: t.List[str]
    lib_source: t.Union[HttpUrl, PathLike]

    def set_custom_index(self, index: str):
        self.pip_index = index

    def set_lib_source(self, source: t.Union[HttpUrl, PathLike]):
        self.lib_source = source

    def retrieve(self, destination: PathLike):
        PackageRetriever.retrieve(self.lib_source, destination)

    def pip_install(self):
        install_command = [sys.executable, '-m', 'pip', 'install']
        if self.pip_index:
            install_command = ["--index-url", self.pip_index]
        install_command += self.packages
        subprocess.check_call(install_command)

class PlatformPackages(BaseModel):
    platform: Platform
    ml_packages: t.Dict[str, MLPackage]

    @classmethod
    def from_json_file(cls, json_file: PathLike) -> _PlatformPackages:
        with open(json_file, "r") as f:
            config_json = json.load(json_file)
        platform = Platform.from_str(**config_json["platform"])
        ml_packages = {
            ml_package["name"]:MLPackage(**ml_package) for ml_package in config_json["ml_packages"]
        }
        return cls(platform, ml_packages)

def load_platform_configs(config_file_path: pathlib.Path) -> t.Dict[Platform, PlatformPackages]:
    configs = {}
    for file in config_file_path.glob("*.json"):
        dependencies = PlatformPackages.from_json_file(file)
        configs[dependencies.platform] = dependencies
    return configs

DEFAULT_MLPACKAGE_PATH = pathlib.Path(
    resources.files(
        "smartsim._core._install.configs.mlpackages").as_file()
    )
DEFAULT_MLPACKAGES = load_platform_configs(DEFAULT_MLPACKAGE_PATH)