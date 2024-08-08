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

import importlib.resources as pkg_resources
from pydantic import HttpUrl
from pydantic.dataclasses import dataclass

import enum
import json
import os
import pathlib
import platform
import textwrap
import typing as t

from .types import PathLike

_PlatformType = t.TypeVar("_PlatformType", bound="Platform")

class PlatformError(Exception):
    pass

class UnsupportedError(PlatformError):
    pass

class PathNotFoundError(PlatformError)
    pass

class Architecture(enum.Enum):
    X64 = "x86_64"
    ARM64 = "arm64"

    @classmethod
    def from_str(cls, string: str, /) -> "Architecture":
        string = string.lower()
        for type_ in cls:
            if string in type_.value:
                return type_
        raise UnsupportedError(f"Unrecognized or unsupported architecture: {string}")

    @classmethod
    def autodetect(cls):
        return cls.from_str(platform.machine())


class Device(enum.Enum):
    CPU = "cpu"
    CUDA118 = "cuda-11.8"
    CUDA121 = "cuda-12.1"
    ROCM57 = "rocm-5.7"

    @classmethod
    def from_str(cls, str_: str) -> "Device":
        str_ = str_.lower()
        if str_ == "gpu":
            # TODO: auto detect which device to use
            #       currently hard coded to `cuda11`
            return cls.CUDA118
        return cls(str_)

    @classmethod
    def detect_cuda_version(cls) -> t.Optional[str]:
        if cuda_path := pathlib.Path(os.environ.get("CUDATOOLKIT_HOME")):
            with open(cuda_path / "version.json", "r") as f:
                cuda_versions = json.load(f)
            major, minor = cuda_versions["cuda"]["version"].split(".")[0:2]
            return cls.from_string(f"cuda-{major}.{minor}")
        return None

    @classmethod
    def detect_rocm_version(cls) -> t.Optional[str]:
        if rocm_path := pathlib.Path(os.environ.get("ROCM_HOME")):
            with open(rocm_path / ".info" / "version", "r") as f:
                major, minor = f.readline().split(".")[0:2]
            return cls.from_string(f"rocm-{major}.{minor}")
        return None

    @classmethod
    def autodetect(cls):
        if device := cls.detect_cuda_version():
            return device
        if device := cls.detect_rocm_Version():
            return device
        return cls.CPU

    def is_gpu(self) -> bool:
        return self != type(self).CPU

    def is_cuda(self) -> bool:
        cls = type(self)
        return self in (cls.CUDA118, cls.CUDA121)

    def is_rocm(self) -> bool:
        cls = type(self)
        return self in (cls.ROCM57)


class OperatingSystem(enum.Enum):
    LINUX = "linux"
    DARWIN = "darwin"

    @classmethod
    def from_str(cls, string: str, /) -> "OperatingSystem":
        string = string.lower()
        for type_ in cls:
            if string in type_.value:
                return type_
        raise UnsupportedError(f"Unrecognized or unsupported operating system: {string}")

    @classmethod
    def autodetect(cls) -> "OperatingSystem":
        return cls.from_str(platform.system())

@dataclass(frozen=True)
class Platform:
    os: OperatingSystem
    architecture: Architecture
    device: Device

    @classmethod
    def from_str(cls, os: str, architecture: str, device: str) -> _Platform:
        return cls(
            OperatingSystem.from_str(os),
            Architecture.from_str(architecture),
            Device.from_str(device)
        )

    def __repr__(self):
        output = [
            self.os.name,
            self.architecture.name,
            self.device.name,
        ]
        return "-".join(output)

@dataclass
class MLBackend:
    name: str
    version: str
    pip_index: str
    python_packages: t.List[str]
    lib_source: t.Union[HttpUrl, _PathLike]

    def set_custom_index(self, index: str):
        self.pip_index = index

    def set_lib_source(self, source: t.Union[HttpUrl, _PathLike]):
        self.lib_source = source

    def __repr__(self):
        output = [
            f"Name: {self.name}",
            f"Version: {self.version}",
            f"Source: {self.lib_source}",
            "Python packages:",
            textwrap.indent("\n".join(self.python_packages), "\t"),
        ]
        return "\n".join(output)


@dataclass
class PlatformDependencies:
    platform: Platform
    ml_backends: t.Dict[str, MLBackend]

    @classmethod
    def from_json_file(cls, json_file: _PathLike) -> _PlatformDependencies:
        with open(json_file, "r") as f:
            config_json = json.load(f)
        platform = Platform.from_str(**config_json["platform"])
        ml_backends = {
            backend["name"]: MLBackend(**backend) for backend in config_json["ml_backends"]
        }
        return cls(platform, ml_backends)

    def __repr__(self):
        output = [
            f"\nPlatform: {str(self.platform)}",
            "ML Packages:",
        ]
        for backend in self.ml_backends.values():
            output += [textwrap.indent(str(backend), "\t")]
            output += ["\n"]

        return "\n".join(output)


def load_platforms(config_file_path: pathlib.Path) -> t.Dict[Platform, PlatformDependencies]:

    configs = {}

    for file in config_file_path.glob("*.json"):
        dependencies = PlatformDependencies.from_json_file(file)
        configs[dependencies.platform] = dependencies
    return configs

DEFAULT_PLATFORM_PATH = pkg_resources.files("smartsim") / "_core" / "_install" / "platforms"
DEFAULT_PLATFORMS = load_platforms(DEFAULT_PLATFORM_PATH)
