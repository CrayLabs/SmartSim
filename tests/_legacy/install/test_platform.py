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
import platform

import pytest

from smartsim._core._install.platform import Architecture, Device, OperatingSystem

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_device_cpu():
    cpu_enum = Device.CPU
    assert not cpu_enum.is_gpu()
    assert not cpu_enum.is_cuda()
    assert not cpu_enum.is_rocm()


@pytest.mark.parametrize("cuda_device", Device.cuda_enums())
def test_cuda(monkeypatch, test_dir, cuda_device):
    version = cuda_device.value.split("-")[1]
    fake_full_version = version + ".8888" ".9999"
    monkeypatch.setenv("CUDA_HOME", test_dir)

    mock_version = dict(cuda=dict(version=fake_full_version))
    print(mock_version)
    with open(f"{test_dir}/version.json", "w") as outfile:
        json.dump(mock_version, outfile)

    assert Device.detect_cuda_version() == cuda_device
    assert cuda_device.is_gpu()
    assert cuda_device.is_cuda()
    assert not cuda_device.is_rocm()


@pytest.mark.parametrize("rocm_device", Device.rocm_enums())
def test_rocm(monkeypatch, test_dir, rocm_device):
    version = rocm_device.value.split("-")[1]
    fake_full_version = version + ".8888" + "-9999"
    monkeypatch.setenv("ROCM_HOME", test_dir)
    info_dir = f"{test_dir}/.info"
    os.mkdir(info_dir)

    with open(f"{info_dir}/version", "w") as outfile:
        outfile.write(fake_full_version)

    assert Device.detect_rocm_version() == rocm_device
    assert rocm_device.is_gpu()
    assert not rocm_device.is_cuda()
    assert rocm_device.is_rocm()


@pytest.mark.parametrize("os", ("linux", "darwin"))
def test_operating_system(monkeypatch, os):
    monkeypatch.setattr(platform, "system", lambda: os)
    assert OperatingSystem.autodetect().value == os


@pytest.mark.parametrize("arch", ("x86_64", "arm64"))
def test_architecture(monkeypatch, arch):
    monkeypatch.setattr(platform, "machine", lambda: arch)
    assert Architecture.autodetect().value == arch
