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


class PlatformError(Exception):
    pass


class UnsupportedError(PlatformError):
    pass


class Architecture(enum.Enum):
    """Identifiers for supported CPU architectures

    :return: An enum representing the CPU architecture
    """

    X64 = "x86_64"
    ARM64 = "arm64"

    @classmethod
    def from_str(cls, string: str) -> "Architecture":
        """Return enum associated with the architecture

        :param string: String representing the architecture, see platform.machine
        :return: Enum for a specific architecture
        """
        string = string.lower()
        return cls(string)

    @classmethod
    def autodetect(cls) -> "Architecture":
        """Automatically return the architecture of the current machine

        :return: enum of this platform's architecture
        """
        return cls.from_str(platform.machine())


class Device(enum.Enum):
    """Identifiers for the device stack

    :return: Enum associated with the device stack
    """

    CPU = "cpu"
    CUDA11 = "cuda-11"
    CUDA12 = "cuda-12"
    ROCM5 = "rocm-5"
    ROCM6 = "rocm-6"

    @classmethod
    def from_str(cls, str_: str) -> "Device":
        """Return enum associated with the device

        :param string: String representing the device and version
        :return: Enum for a specific device
        """
        str_ = str_.lower()
        if str_ == "gpu":
            # TODO: auto detect which device to use
            #       currently hard coded to `cuda11`
            return cls.CUDA11
        return cls(str_)

    @classmethod
    def detect_cuda_version(cls) -> t.Optional["Device"]:
        """Find the enum based on environment CUDA

        :return: Enum for the version of CUDA currently available
        """
        if cuda_home := os.environ.get("CUDA_HOME"):
            cuda_path = pathlib.Path(cuda_home)
            with open(cuda_path / "version.json", "r", encoding="utf-8") as file_handle:
                cuda_versions = json.load(file_handle)
            major = cuda_versions["cuda"]["version"].split(".")[0]
            return cls.from_str(f"cuda-{major}")
        return None

    @classmethod
    def detect_rocm_version(cls) -> t.Optional["Device"]:
        """Find the enum based on environment ROCm

        :return: Enum for the version of ROCm currently available
        """
        if rocm_home := os.environ.get("ROCM_HOME"):
            rocm_path = pathlib.Path(rocm_home)
            fname = rocm_path / ".info" / "version"
            with open(fname, "r", encoding="utf-8") as file_handle:
                major = file_handle.readline().split("-")[0].split(".")[0]
            return cls.from_str(f"rocm-{major}")
        return None

    def is_gpu(self) -> bool:
        """Whether the enum is categorized as a GPU

        :return: True if GPU
        """
        return self != type(self).CPU

    def is_cuda(self) -> bool:
        """Whether the enum is associated with a CUDA device

        :return: True for any supported CUDA enums
        """
        cls = type(self)
        return self in cls.cuda_enums()

    def is_rocm(self) -> bool:
        """Whether the enum is associated with a ROCm device

        :return: True for any supported ROCm enums
        """
        cls = type(self)
        return self in cls.rocm_enums()

    @classmethod
    def cuda_enums(cls) -> t.Tuple["Device", ...]:
        """Detect all CUDA devices supported by SmartSim

        :return: all enums associated with CUDA
        """
        return tuple(device for device in cls if "cuda" in device.value)

    @classmethod
    def rocm_enums(cls) -> t.Tuple["Device", ...]:
        """Detect all ROCm devices supported by SmartSim

        :return: all enums associated with ROCm
        """
        return tuple(device for device in cls if "rocm" in device.value)


class OperatingSystem(enum.Enum):
    """Enum for all supported operating systems"""

    LINUX = "linux"
    DARWIN = "darwin"

    @classmethod
    def from_str(cls, string: str, /) -> "OperatingSystem":
        """Return enum associated with the OS

        :param string: String representing the OS
        :return: Enum for a specific OS
        """
        string = string.lower()
        return cls(string)

    @classmethod
    def autodetect(cls) -> "OperatingSystem":
        """Automatically return the OS of the current machine

        :return: enum of this platform's OS
        """
        return cls.from_str(platform.system())


@dataclass(frozen=True)
class Platform:
    """Container describing relevant identifiers for a platform"""

    operating_system: OperatingSystem
    architecture: Architecture
    device: Device

    @classmethod
    def from_strs(cls, operating_system: str, architecture: str, device: str) -> Self:
        """Factory method for Platform from string onput

        :param os: String identifier for the OS
        :param architecture: String identifier for the architecture
        :param device: String identifer for the device and version
        :return: Instance of Platform
        """
        return cls(
            OperatingSystem.from_str(operating_system),
            Architecture.from_str(architecture),
            Device.from_str(device),
        )

    def __str__(self) -> str:
        """Human-readable representation of Platform

        :return: String created from the values of the enums for each property
        """
        output = [
            self.operating_system.name,
            self.architecture.name,
            self.device.name,
        ]
        return "-".join(output)
