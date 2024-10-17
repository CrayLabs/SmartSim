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

from pathlib import Path

import pytest

from smartsim._core._install.buildenv import BuildEnv
from smartsim._core._install.mlpackages import (
    DEFAULT_MLPACKAGE_PATH,
    MLPackage,
    load_platform_configs,
)
from smartsim._core._install.platform import Platform
from smartsim._core._install.redisaiBuilder import RedisAIBuilder

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a

DEFAULT_MLPACKAGES = load_platform_configs(DEFAULT_MLPACKAGE_PATH)


@pytest.mark.parametrize(
    "platform",
    [platform for platform in DEFAULT_MLPACKAGES],
    ids=[str(platform) for platform in DEFAULT_MLPACKAGES],
)
def test_backends_to_be_installed(monkeypatch, test_dir, platform):
    mlpackages = DEFAULT_MLPACKAGES[platform]
    monkeypatch.setattr(MLPackage, "retrieve", lambda *args, **kwargs: None)
    builder = RedisAIBuilder(platform, mlpackages, BuildEnv(), Path(test_dir))

    BACKENDS = ["libtorch", "libtensorflow", "onnxruntime"]
    TOGGLES = ["build_torch", "build_tensorflow", "build_onnxruntime"]

    for backend, toggle in zip(BACKENDS, TOGGLES):
        assert getattr(builder, toggle) == (backend in mlpackages)
