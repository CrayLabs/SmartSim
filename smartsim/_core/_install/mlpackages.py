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

import json
import os
import pathlib
import re
import subprocess
import sys
import typing as t
from collections.abc import MutableMapping
from dataclasses import dataclass

from tabulate import tabulate

from .platform import Platform
from .types import PathLike
from .utils import retrieve


class RequireRelativePath(Exception):
    pass


@dataclass
class RAIPatch:
    """Holds information about how to patch a RedisAI source file

    :param description: Human-readable description of the patch's purpose
    :param replacement: "The replacement for the line found by the regex"
    :param source_file: A relative path to the chosen file
    :param regex: A regex pattern to match in the given file

    """

    description: str
    replacement: str
    source_file: pathlib.Path
    regex: re.Pattern[str]

    def __post_init__(self) -> None:
        self.source_file = pathlib.Path(self.source_file)
        self.regex = re.compile(self.regex)


@dataclass
class MLPackage:
    """Describes the python and C/C++ library for an ML package"""

    name: str
    version: str
    pip_index: str
    python_packages: t.List[str]
    lib_source: PathLike
    rai_patches: t.Tuple[RAIPatch, ...] = ()

    def retrieve(self, destination: PathLike) -> None:
        """Retrieve an archive and/or repository for the package

        :param destination: Path to place the extracted package or repository
        """
        retrieve(self.lib_source, pathlib.Path(destination))

    def pip_install(self, quiet: bool = False) -> None:
        """Install associated python packages

        :param quiet: If True, suppress most of the pip output, defaults to False
        """
        if self.python_packages:
            install_command = [sys.executable, "-m", "pip", "install"]
            if self.pip_index:
                install_command += ["--index-url", self.pip_index]
            if quiet:
                install_command += ["--quiet", "--no-warn-conflicts"]
            install_command += self.python_packages
            subprocess.check_call(install_command)


class MLPackageCollection(MutableMapping[str, MLPackage]):
    """Collects multiple MLPackages

    Define a collection of MLPackages available for a specific platform
    """

    def __init__(self, platform: Platform, ml_packages: t.Sequence[MLPackage]):
        self.platform = platform
        self._ml_packages = {pkg.name: pkg for pkg in ml_packages}

    @classmethod
    def from_json_file(cls, json_file: PathLike) -> "MLPackageCollection":
        """Create an MLPackageCollection specified from a JSON file

        :param json_file: path to the JSON file
        :return: An instance of MLPackageCollection for a platform
        """
        with open(json_file, "r", encoding="utf-8") as file_handle:
            config_json = json.load(file_handle)
        platform = Platform.from_strs(**config_json["platform"])

        for ml_package in config_json["ml_packages"]:
            # Convert the dictionary representation to a RAIPatch
            if "rai_patches" in ml_package:
                patch_list = ml_package.pop("rai_patches")
                ml_package["rai_patches"] = [RAIPatch(**patch) for patch in patch_list]

        ml_packages = [
            MLPackage(**ml_package) for ml_package in config_json["ml_packages"]
        ]
        return cls(platform, ml_packages)

    def __iter__(self) -> t.Iterator[str]:
        """Iterate over the mlpackages in the collection

        :return: Iterator over mlpackages
        """
        return iter(self._ml_packages)

    def __getitem__(self, key: str) -> MLPackage:
        """Retrieve an MLPackage based on its name

        :param key: Name of the python package (e.g. libtorch)
        :return: MLPackage with all requirements
        """
        return self._ml_packages[key]

    def __len__(self) -> int:
        return len(self._ml_packages)

    def __delitem__(self, key: str) -> None:
        del self._ml_packages[key]

    def __setitem__(self, key: t.Any, value: t.Any) -> t.NoReturn:
        raise TypeError(f"{type(self).__name__} does not support item assignment")

    def __contains__(self, key: object) -> bool:
        return key in self._ml_packages

    def __str__(self, tablefmt: str = "github") -> str:
        """Display package names and versions as a table

        :param tablefmt: Tabulate format, defaults to "github"
        """

        return tabulate(
            [[k, v.version] for k, v in self._ml_packages.items()],
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
    if not config_file_path.is_dir():
        path = os.fspath(config_file_path)
        msg = f"Platform configuration directory `{path}` does not exist"
        raise FileNotFoundError(msg)
    configs = {}
    for config_file in config_file_path.glob("*.json"):
        dependencies = MLPackageCollection.from_json_file(config_file)
        configs[dependencies.platform] = dependencies
    return configs


DEFAULT_MLPACKAGE_PATH: t.Final = (
    pathlib.Path(__file__).parent / "configs" / "mlpackages"
)
DEFAULT_MLPACKAGES: t.Final = load_platform_configs(DEFAULT_MLPACKAGE_PATH)
