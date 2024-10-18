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

import logging

import pytest

from smartsim.settings import LaunchSettings
from smartsim.settings.launch_command import LauncherType

pytestmark = pytest.mark.group_a


@pytest.mark.parametrize(
    "launch_enum",
    [pytest.param(type_, id=type_.value) for type_ in LauncherType],
)
def test_create_launch_settings(launch_enum):
    ls_str = LaunchSettings(
        launcher=launch_enum.value,
        launch_args={"launch": "var"},
        env_vars={"ENV": "VAR"},
    )
    assert ls_str._launcher == launch_enum
    # TODO need to test launch_args
    assert ls_str._env_vars == {"ENV": "VAR"}

    ls_enum = LaunchSettings(
        launcher=launch_enum, launch_args={"launch": "var"}, env_vars={"ENV": "VAR"}
    )
    assert ls_enum._launcher == launch_enum
    # TODO need to test launch_args
    assert ls_enum._env_vars == {"ENV": "VAR"}


def test_launcher_property():
    ls = LaunchSettings(launcher="local")
    assert ls.launcher == "local"


def test_env_vars_property():
    ls = LaunchSettings(launcher="local", env_vars={"ENV": "VAR"})
    assert ls.env_vars == {"ENV": "VAR"}
    ref = ls.env_vars
    assert ref is ls.env_vars


def test_update_env_vars():
    ls = LaunchSettings(launcher="local", env_vars={"ENV": "VAR"})
    ls.update_env({"test": "no_update"})
    assert ls.env_vars == {"ENV": "VAR", "test": "no_update"}


def test_update_env_vars_errors():
    ls = LaunchSettings(launcher="local", env_vars={"ENV": "VAR"})
    with pytest.raises(TypeError):
        ls.update_env({"test": 1})
    with pytest.raises(TypeError):
        ls.update_env({1: "test"})
    with pytest.raises(TypeError):
        ls.update_env({1: 1})
    with pytest.raises(TypeError):
        # Make sure the first key and value do not assign
        # and that the function is atomic
        ls.update_env({"test": "test", "test": 1})
        assert ls.env_vars == {"ENV": "VAR"}
