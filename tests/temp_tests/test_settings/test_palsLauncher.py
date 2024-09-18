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
from smartsim.settings.arguments.launch.pals import (
    PalsMpiexecLaunchArguments,
    _as_pals_command,
)
from smartsim.settings.launch_command import LauncherType

pytestmark = pytest.mark.group_a


def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    ls = LaunchSettings(launcher=LauncherType.Pals)
    assert ls.launch_args.launcher_str() == LauncherType.Pals.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param(
            "set_cpu_binding_type",
            ("bind",),
            "bind",
            "bind-to",
            id="set_cpu_binding_type",
        ),
        pytest.param("set_tasks", (2,), "2", "np", id="set_tasks"),
        pytest.param("set_tasks_per_node", (2,), "2", "ppn", id="set_tasks_per_node"),
        pytest.param(
            "set_hostlist", ("host_A",), "host_A", "hosts", id="set_hostlist_str"
        ),
        pytest.param(
            "set_hostlist",
            (["host_A", "host_B"],),
            "host_A,host_B",
            "hosts",
            id="set_hostlist_list[str]",
        ),
        pytest.param(
            "set_executable_broadcast",
            ("broadcast",),
            "broadcast",
            "transfer",
            id="set_executable_broadcast",
        ),
    ],
)
def test_pals_class_methods(function, value, flag, result):
    palsLauncher = LaunchSettings(launcher=LauncherType.Pals)
    assert isinstance(palsLauncher.launch_args, PalsMpiexecLaunchArguments)
    getattr(palsLauncher.launch_args, function)(*value)
    assert palsLauncher.launch_args._launch_args[flag] == result
    assert palsLauncher._arguments.format_launch_args() == ["--" + flag, str(result)]


def test_format_env_vars():
    env_vars = {"FOO_VERSION": "3.14", "PATH": None, "LD_LIBRARY_PATH": None}
    palsLauncher = LaunchSettings(launcher=LauncherType.Pals, env_vars=env_vars)
    formatted = " ".join(palsLauncher._arguments.format_env_vars(env_vars))
    expected = "--env FOO_VERSION=3.14 --envlist PATH,LD_LIBRARY_PATH"
    assert formatted == expected


def test_invalid_hostlist_format():
    """Test invalid hostlist formats"""
    palsLauncher = LaunchSettings(launcher=LauncherType.Pals)
    with pytest.raises(TypeError):
        palsLauncher.launch_args.set_hostlist(["test", 5])
    with pytest.raises(TypeError):
        palsLauncher.launch_args.set_hostlist([5])
    with pytest.raises(TypeError):
        palsLauncher.launch_args.set_hostlist(5)


@pytest.mark.parametrize(
    "args, expected",
    (
        pytest.param({}, ("mpiexec", "--", "echo", "hello", "world"), id="Empty Args"),
        pytest.param(
            {"n": "1"},
            ("mpiexec", "--n", "1", "--", "echo", "hello", "world"),
            id="Short Arg",
        ),
        pytest.param(
            {"host": "myhost"},
            ("mpiexec", "--host", "myhost", "--", "echo", "hello", "world"),
            id="Long Arg",
        ),
        pytest.param(
            {"v": None},
            ("mpiexec", "--v", "--", "echo", "hello", "world"),
            id="Short Arg (No Value)",
        ),
        pytest.param(
            {"verbose": None},
            ("mpiexec", "--verbose", "--", "echo", "hello", "world"),
            id="Long Arg (No Value)",
        ),
        pytest.param(
            {"n": "1", "host": "myhost"},
            ("mpiexec", "--n", "1", "--host", "myhost", "--", "echo", "hello", "world"),
            id="Short and Long Args",
        ),
    ),
)
def test_formatting_launch_args(args, expected, test_dir):
    out = os.path.join(test_dir, "out.txt")
    err = os.path.join(test_dir, "err.txt")
    open(out, "w"), open(err, "w")
    shell_launch_cmd = _as_pals_command(
        PalsMpiexecLaunchArguments(args),
        ("echo", "hello", "world"),
        test_dir,
        {},
        out,
        err,
    )
    assert isinstance(shell_launch_cmd, ShellLauncherCommand)
    assert shell_launch_cmd.command_tuple == expected
    assert shell_launch_cmd.path == pathlib.Path(test_dir)
    assert shell_launch_cmd.env == {}
    assert isinstance(shell_launch_cmd.stdout, io.TextIOWrapper)
    assert shell_launch_cmd.stdout.name == out
    assert isinstance(shell_launch_cmd.stderr, io.TextIOWrapper)
    assert shell_launch_cmd.stderr.name == err
