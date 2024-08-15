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

from .platform import Platform
from .types import PathLike
from .utils import PackageRetriever


@dataclass
class MLPackage:
    """Describes the python and C/C++ library for an ML package"""

    name: str
    version: str
    pip_index: str
    python_packages: t.List[str]
    lib_source: PathLike

    def retrieve(self, destination: PathLike) -> None:
        """Retrieve an archive and/or repository for the package

        :param destination: Path to place the extracted package or repository
        """
        PackageRetriever.retrieve(self.lib_source, pathlib.Path(destination))

    def pip_install(self, quiet: bool = False) -> None:
        """Install associated python packages

        :param quiet: If True, suppress most of the pip output, defaults to False
        """
        if self.python_packages:
            install_command = [sys.executable, "-m", "pip", "install"]
            if self.pip_index:
                install_command += ["--index-url", self.pip_index]
            if quiet:
                install_command += ["--quiet"]
            install_command += self.python_packages
            subprocess.check_call(install_command)


@dataclass
class MLPackageCollection:
    """Collects multiple MLPackages

    Define a collection of MLPackages available for a specific platform
    """

    platform: Platform
    ml_packages: t.Dict[str, MLPackage]

    @classmethod
    def from_json_file(cls, json_file: PathLike) -> "MLPackageCollection":
        """Create an MLPackageCollection specified from a JSON file

        :param json_file: path to the JSON file
        :return: An instance of MLPackageCollection for a platform
        """
        with open(json_file, "r") as f:
            config_json = json.load(f)
        platform = Platform.from_str(**config_json["platform"])
        ml_packages = {
            ml_package["name"]: MLPackage(**ml_package)
            for ml_package in config_json["ml_packages"]
        }
        return cls(platform, ml_packages)

    def __iter__(self) -> t.Iterator[str]:
        """Iterate over the mlpackages in the collection

        :return: Iterator over mlpackages
        """
        return iter(self.ml_packages)

    def __getitem__(self, key: str) -> MLPackage:
        """Retrieve an MLPackage based on its name

        :param key: Name of the python package (e.g. libtorch)
        :return: MLPackage with all requirements
        """
        return self.ml_packages[key]

    def values(self) -> t.Iterable[MLPackage]:
        """Accesses the MLPackages directly

        :return: All the MLPackages in this collection
        """
        return self.ml_packages.values()

    def items(self) -> t.ItemsView[str, MLPackage]:
        """Retrieve all MLPackages and their names

        :return: MLPackages and names in this collection
        """
        return self.ml_packages.items()

    def keys(self) -> t.Iterable[str]:
        """Retrieve just the names of the MLPackages

        :return: The names of the MLPackages
        """
        return self.ml_packages.keys()

    def pop(self, key: str) -> None:
        """Remove a particular MLPackage by name

        Used internally if a user specifies they do not want a
        particular package installed

        :param key: The name of the package to remove
        """
        self.ml_packages.pop(key)

    def tabulate_versions(self, tablefmt: str = "github") -> str:
        """Display package names and versions as a table

        :param tablefmt: Tabulate format, defaults to "github"
        :type tablefmt: str, optional
        """

        return tabulate(
            [[k, v.version] for k, v in self.items()],
            headers=["Package", "Version"],
            tablefmt=tablefmt,
        )


def load_platform_configs(
    config_file_path: pathlib.Path,
) -> t.Dict[Platform, MLPackageCollection]:
    """Create MLPackageCollections from JSON files in directory

    :param config_file_path: Directory with JSON files describing the
                             configuration by platform
    :return: Dictionary whose keys are the supported platform and values
             are its associated MLPackageCollection
    """
    configs = {}
    for config_file in config_file_path.glob("*.json"):
        dependencies = MLPackageCollection.from_json_file(config_file)
        configs[dependencies.platform] = dependencies
    return configs


DEFAULT_MLPACKAGE_PATH: pathlib.Path = pathlib.Path(
    str(resources.files("smartsim._core._install.configs.mlpackages").joinpath(""))
)
DEFAULT_MLPACKAGES: t.Dict[Platform, MLPackageCollection] = load_platform_configs(
    DEFAULT_MLPACKAGE_PATH
)
