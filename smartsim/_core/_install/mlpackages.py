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

from dataclasses import dataclass

from tabulate import tabulate

from .types import PathLike
from .utils import PackageRetriever
from .platform import Platform

_MLPackageCollection = t.TypeVar("_MLPackageCollection", bound="MLPackageCollection")

@dataclass
class MLPackage():
    name: str
    version: str
    pip_index: str
    python_packages: t.List[str]
    lib_source:PathLike

    def set_custom_index(self, index: str):
        self.pip_index = index

    def set_lib_source(self, source: PathLike):
        self.lib_source = source

    def retrieve(self, destination: PathLike):
        PackageRetriever.retrieve(self.lib_source, destination)

    def pip_install(self, quiet: bool = False):
        if self.python_packages:
            install_command = [sys.executable, '-m', 'pip', 'install']
            if self.pip_index:
                install_command += ["--index-url", self.pip_index]
            if quiet:
                install_command += ["--quiet"]
            install_command += self.python_packages
            subprocess.check_call(install_command)

@dataclass
class MLPackageCollection():
    platform: Platform
    ml_packages: t.Dict[str, MLPackage]

    @classmethod
    def from_json_file(cls, json_file: PathLike) -> _MLPackageCollection:
        with open(json_file, "r") as f:
            config_json = json.load(f)
        platform = Platform.from_str(**config_json["platform"])
        ml_packages = {
            ml_package["name"]:MLPackage(**ml_package) for ml_package in config_json["ml_packages"]
        }
        return cls(platform, ml_packages)

    def __iter__(self):
        return iter(self.ml_packages)

    def __getitem__(self, key):
        return self.ml_packages[key]

    def values(self):
        return self.ml_packages.values()

    def items(self):
        return self.ml_packages.items()

    def keys(self):
        return self.ml_packages.keys()

    def pop(self, key):
        self.ml_packages.pop(key)

    def tabulate_versions(self, tablefmt: str="github") -> None:
        """Display package names and versions as a table

        :param tablefmt: Tabulate format, defaults to "github"
        :type tablefmt: str, optional
        """

        return tabulate(
            [[k, v.version] for k, v in self.items()],
            headers=["Package", "Version"],
            tablefmt=tablefmt
        )

def load_platform_configs(config_file_path: pathlib.Path) -> t.Dict[Platform, MLPackageCollection]:
    configs = {}
    for config_file in config_file_path.glob("*.json"):
        dependencies = MLPackageCollection.from_json_file(config_file)
        configs[dependencies.platform] = dependencies
    return configs

DEFAULT_MLPACKAGE_PATH: pathlib.Path = pathlib.Path(
    resources.path(
        "smartsim._core._install.configs", "mlpackages")
    )
DEFAULT_MLPACKAGES: t.Dict[Platform, MLPackageCollection] = load_platform_configs(DEFAULT_MLPACKAGE_PATH)
