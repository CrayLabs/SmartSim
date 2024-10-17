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
import subprocess

import pytest

from smartsim.settings import LaunchSettings
from smartsim.settings.arguments.launch.lsf import (
    JsrunLaunchArguments,
    _as_jsrun_command,
)
from smartsim.settings.launch_command import LauncherType

pytestmark = pytest.mark.group_a


def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    ls = LaunchSettings(launcher=LauncherType.Lsf)
    assert ls.launch_args.launcher_str() == LauncherType.Lsf.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_tasks", (2,), "2", "np", id="set_tasks"),
        pytest.param(
            "set_binding", ("packed:21",), "packed:21", "bind", id="set_binding"
        ),
    ],
)
def test_lsf_class_methods(function, value, flag, result):
    lsfLauncher = LaunchSettings(launcher=LauncherType.Lsf)
    assert isinstance(lsfLauncher._arguments, JsrunLaunchArguments)
    getattr(lsfLauncher.launch_args, function)(*value)
    assert lsfLauncher.launch_args._launch_args[flag] == result


def test_format_env_vars():
    env_vars = {"OMP_NUM_THREADS": None, "LOGGING": "verbose"}
    lsfLauncher = LaunchSettings(launcher=LauncherType.Lsf, env_vars=env_vars)
    assert isinstance(lsfLauncher._arguments, JsrunLaunchArguments)
    formatted = lsfLauncher._arguments.format_env_vars(env_vars)
    assert formatted == ["-E", "OMP_NUM_THREADS", "-E", "LOGGING=verbose"]


def test_launch_args():
    """Test the possible user overrides through run_args"""
    launch_args = {
        "latency_priority": "gpu-gpu",
        "immediate": None,
        "d": "packed",  # test single letter variables
        "nrs": 10,
        "np": 100,
    }
    lsfLauncher = LaunchSettings(launcher=LauncherType.Lsf, launch_args=launch_args)
    assert isinstance(lsfLauncher._arguments, JsrunLaunchArguments)
    formatted = lsfLauncher._arguments.format_launch_args()
    result = [
        "--latency_priority=gpu-gpu",
        "--immediate",
        "-d",
        "packed",
        "--nrs=10",
        "--np=100",
    ]
    assert formatted == result


@pytest.mark.parametrize(
    "args, expected",
    (
        pytest.param(
            {},
            (
                "jsrun",
                "--stdio_stdout=output.txt",
                "--stdio_stderr=error.txt",
                "--",
                "echo",
                "hello",
                "world",
            ),
            id="Empty Args",
        ),
        pytest.param(
            {"n": "1"},
            (
                "jsrun",
                "-n",
                "1",
                "--stdio_stdout=output.txt",
                "--stdio_stderr=error.txt",
                "--",
                "echo",
                "hello",
                "world",
            ),
            id="Short Arg",
        ),
        pytest.param(
            {"nrs": "1"},
            (
                "jsrun",
                "--nrs=1",
                "--stdio_stdout=output.txt",
                "--stdio_stderr=error.txt",
                "--",
                "echo",
                "hello",
                "world",
            ),
            id="Long Arg",
        ),
        pytest.param(
            {"v": None},
            (
                "jsrun",
                "-v",
                "--stdio_stdout=output.txt",
                "--stdio_stderr=error.txt",
                "--",
                "echo",
                "hello",
                "world",
            ),
            id="Short Arg (No Value)",
        ),
        pytest.param(
            {"verbose": None},
            (
                "jsrun",
                "--verbose",
                "--stdio_stdout=output.txt",
                "--stdio_stderr=error.txt",
                "--",
                "echo",
                "hello",
                "world",
            ),
            id="Long Arg (No Value)",
        ),
        pytest.param(
            {"tasks_per_rs": "1", "n": "123"},
            (
                "jsrun",
                "--tasks_per_rs=1",
                "-n",
                "123",
                "--stdio_stdout=output.txt",
                "--stdio_stderr=error.txt",
                "--",
                "echo",
                "hello",
                "world",
            ),
            id="Short and Long Args",
        ),
    ),
)
def test_formatting_launch_args(args, expected, test_dir):
    outfile = "output.txt"
    errfile = "error.txt"
    env, path, stdin, stdout, args = _as_jsrun_command(
        JsrunLaunchArguments(args),
        ("echo", "hello", "world"),
        test_dir,
        {},
        outfile,
        errfile,
    )
    assert tuple(args) == expected
    assert path == test_dir
    assert env == {}
    assert stdin == subprocess.DEVNULL
    assert stdout == subprocess.DEVNULL
