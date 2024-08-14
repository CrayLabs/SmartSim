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

import enum
import json
import os
import pathlib
import platform
import typing as t
from dataclasses import dataclass

from typing_extensions import Self

from .types import PathLike


class PlatformError(Exception):
    pass


class UnsupportedError(PlatformError):
    pass


class PathNotFoundError(PlatformError):
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
    def autodetect(cls) -> "Architecture":
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
    def detect_cuda_version(cls) -> t.Optional["Device"]:
        if cuda_home := os.environ.get("CUDA_HOME"):
            cuda_path = pathlib.Path(cuda_home)
            with open(cuda_path / "version.json", "r") as f:
                cuda_versions = json.load(f)
            major, minor = cuda_versions["cuda"]["version"].split(".")[0:2]
            return cls.from_str(f"cuda-{major}.{minor}")
        return None

    @classmethod
    def detect_rocm_version(cls) -> t.Optional["Device"]:
        if rocm_home := os.environ.get("ROCM_HOME"):
            rocm_path = pathlib.Path(rocm_home)
            with open(rocm_path / ".info" / "version", "r") as f:
                major, minor = f.readline().split("-")[0].split(".")
            return cls.from_str(f"rocm-{major}.{minor}")
        return None

    def is_gpu(self) -> bool:
        return self != type(self).CPU

    def is_cuda(self) -> bool:
        cls = type(self)
        return self in (cls.CUDA118, cls.CUDA121)

    def is_rocm(self) -> bool:
        cls = type(self)
        return self in (cls.ROCM57,)

    @classmethod
    def _cuda_enums(cls) -> t.Tuple["Device", ...]:
        return tuple(device for device in cls if "cuda" in device.value)

    @classmethod
    def _rocm_enums(cls) -> t.Tuple["Device", ...]:
        return tuple(device for device in cls if "rocm" in device.value)


class OperatingSystem(enum.Enum):
    LINUX = "linux"
    DARWIN = "darwin"

    @classmethod
    def from_str(cls, string: str, /) -> "OperatingSystem":
        string = string.lower()
        for type_ in cls:
            if string in type_.value:
                return type_
        raise UnsupportedError(
            f"Unrecognized or unsupported operating system: {string}"
        )

    @classmethod
    def autodetect(cls) -> "OperatingSystem":
        return cls.from_str(platform.system())


@dataclass(frozen=True)
class Platform:
    os: OperatingSystem
    architecture: Architecture
    device: Device

    @classmethod
    def from_str(cls, os: str, architecture: str, device: str) -> Self:
        return cls(
            OperatingSystem.from_str(os),
            Architecture.from_str(architecture),
            Device.from_str(device),
        )

    def __repr__(self) -> str:
        output = [
            self.os.name,
            self.architecture.name,
            self.device.name,
        ]
        return "-".join(output)
