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
import io
import os
import pathlib

import pytest

from smartsim._core.shell.shell_launcher import ShellLauncherCommand
from smartsim.settings import LaunchSettings
from smartsim.settings.arguments.launch.local import (
    LocalLaunchArguments,
    _as_local_command,
)
from smartsim.settings.launch_command import LauncherType

pytestmark = pytest.mark.group_a


def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    ls = LaunchSettings(launcher=LauncherType.Local)
    assert ls.launch_args.launcher_str() == LauncherType.Local.value


# TODO complete after launch args retrieval
def test_launch_args_input_mutation():
    # Tests that the run args passed in are not modified after initialization
    key0, key1, key2 = "arg0", "arg1", "arg2"
    val0, val1, val2 = "val0", "val1", "val2"

    default_launcher_args = {
        key0: val0,
        key1: val1,
        key2: val2,
    }
    localLauncher = LaunchSettings(
        launcher=LauncherType.Local, launch_args=default_launcher_args
    )

    # Confirm initial values are set
    assert localLauncher.launch_args._launch_args[key0] == val0
    assert localLauncher.launch_args._launch_args[key1] == val1
    assert localLauncher.launch_args._launch_args[key2] == val2

    # Update our common run arguments
    val2_upd = f"not-{val2}"
    default_launcher_args[key2] = val2_upd

    # Confirm previously created run settings are not changed
    assert localLauncher.launch_args._launch_args[key2] == val2


@pytest.mark.parametrize(
    "env_vars",
    [
        pytest.param({}, id="no env vars"),
        pytest.param({"env1": "abc"}, id="normal var"),
        pytest.param({"env1": "abc,def"}, id="compound var"),
        pytest.param({"env1": "xyz", "env2": "pqr"}, id="multiple env vars"),
    ],
)
def test_update_env(env_vars):
    """Ensure non-initialized env vars update correctly"""
    localLauncher = LaunchSettings(launcher=LauncherType.Local)
    localLauncher.update_env(env_vars)

    assert len(localLauncher.env_vars) == len(env_vars.keys())


def test_format_launch_args():
    localLauncher = LaunchSettings(launcher=LauncherType.Local, launch_args={"-np": 2})
    launch_args = localLauncher._arguments.format_launch_args()
    assert launch_args == ["-np", "2"]


@pytest.mark.parametrize(
    "env_vars",
    [
        pytest.param({"env1": {"abc"}}, id="set value not allowed"),
        pytest.param({"env1": {"abc": "def"}}, id="dict value not allowed"),
    ],
)
def test_update_env_null_valued(env_vars):
    """Ensure validation of env var in update"""
    orig_env = {}

    with pytest.raises(TypeError) as ex:
        localLauncher = LaunchSettings(launcher=LauncherType.Local, env_vars=orig_env)
        localLauncher.update_env(env_vars)


@pytest.mark.parametrize(
    "env_vars",
    [
        pytest.param({}, id="no env vars"),
        pytest.param({"env1": "abc"}, id="normal var"),
        pytest.param({"env1": "abc,def"}, id="compound var"),
        pytest.param({"env1": "xyz", "env2": "pqr"}, id="multiple env vars"),
    ],
)
def test_update_env_initialized(env_vars):
    """Ensure update of initialized env vars does not overwrite"""
    orig_env = {"key": "value"}
    localLauncher = LaunchSettings(launcher=LauncherType.Local, env_vars=orig_env)
    localLauncher.update_env(env_vars)

    combined_keys = {k for k in env_vars.keys()}
    combined_keys.update(k for k in orig_env.keys())

    assert len(localLauncher.env_vars) == len(combined_keys)
    assert {k for k in localLauncher.env_vars.keys()} == combined_keys


def test_format_env_vars():
    env_vars = {
        "A": "a",
        "B": None,
        "C": "",
        "D": "12",
    }
    localLauncher = LaunchSettings(launcher=LauncherType.Local, env_vars=env_vars)
    assert isinstance(localLauncher._arguments, LocalLaunchArguments)
    assert localLauncher._arguments.format_env_vars(env_vars) == [
        "A=a",
        "B=",
        "C=",
        "D=12",
    ]


def test_formatting_returns_original_exe(test_dir):
    out = os.path.join(test_dir, "out.txt")
    err = os.path.join(test_dir, "err.txt")
    open(out, "w"), open(err, "w")
    shell_launch_cmd = _as_local_command(
        LocalLaunchArguments({}), ("echo", "hello", "world"), test_dir, {}, out, err
    )
    assert isinstance(shell_launch_cmd, ShellLauncherCommand)
    assert shell_launch_cmd.command_tuple == ("echo", "hello", "world")
    assert shell_launch_cmd.path == pathlib.Path(test_dir)
    assert shell_launch_cmd.env == {}
    assert isinstance(shell_launch_cmd.stdout, io.TextIOWrapper)
    assert shell_launch_cmd.stdout.name == out
    assert isinstance(shell_launch_cmd.stderr, io.TextIOWrapper)
    assert shell_launch_cmd.stderr.name == err
