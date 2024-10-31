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
from smartsim.settings.arguments.launch.alps import (
    AprunLaunchArguments,
    _as_aprun_command,
)
from smartsim.settings.launch_command import LauncherType

pytestmark = pytest.mark.group_a


def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    alpsLauncher = LaunchSettings(launcher=LauncherType.Alps)
    assert alpsLauncher.launch_args.launcher_str() == LauncherType.Alps.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param(
            "set_cpus_per_task", (4,), "4", "cpus-per-pe", id="set_cpus_per_task"
        ),
        pytest.param("set_tasks", (4,), "4", "pes", id="set_tasks"),
        pytest.param(
            "set_tasks_per_node", (4,), "4", "pes-per-node", id="set_tasks_per_node"
        ),
        pytest.param(
            "set_hostlist", ("host_A",), "host_A", "node-list", id="set_hostlist_str"
        ),
        pytest.param(
            "set_hostlist",
            (["host_A", "host_B"],),
            "host_A,host_B",
            "node-list",
            id="set_hostlist_list[str]",
        ),
        pytest.param(
            "set_hostlist_from_file",
            ("./path/to/hostfile",),
            "./path/to/hostfile",
            "node-list-file",
            id="set_hostlist_from_file",
        ),
        pytest.param(
            "set_excluded_hosts",
            ("host_A",),
            "host_A",
            "exclude-node-list",
            id="set_excluded_hosts_str",
        ),
        pytest.param(
            "set_excluded_hosts",
            (["host_A", "host_B"],),
            "host_A,host_B",
            "exclude-node-list",
            id="set_excluded_hosts_list[str]",
        ),
        pytest.param(
            "set_cpu_bindings", (4,), "4", "cpu-binding", id="set_cpu_bindings"
        ),
        pytest.param(
            "set_cpu_bindings",
            ([4, 4],),
            "4,4",
            "cpu-binding",
            id="set_cpu_bindings_list[str]",
        ),
        pytest.param(
            "set_memory_per_node",
            (8000,),
            "8000",
            "memory-per-pe",
            id="set_memory_per_node",
        ),
        pytest.param(
            "set_walltime",
            ("10:00:00",),
            "10:00:00",
            "cpu-time-limit",
            id="set_walltime",
        ),
        pytest.param(
            "set_verbose_launch", (True,), "7", "debug", id="set_verbose_launch"
        ),
        pytest.param("set_quiet_launch", (True,), None, "quiet", id="set_quiet_launch"),
    ],
)
def test_alps_class_methods(function, value, flag, result):
    alpsLauncher = LaunchSettings(launcher=LauncherType.Alps)
    assert isinstance(alpsLauncher._arguments, AprunLaunchArguments)
    getattr(alpsLauncher.launch_args, function)(*value)
    assert alpsLauncher.launch_args._launch_args[flag] == result


def test_set_verbose_launch():
    alpsLauncher = LaunchSettings(launcher=LauncherType.Alps)
    assert isinstance(alpsLauncher._arguments, AprunLaunchArguments)
    alpsLauncher.launch_args.set_verbose_launch(True)
    assert alpsLauncher.launch_args._launch_args == {"debug": "7"}
    alpsLauncher.launch_args.set_verbose_launch(False)
    assert alpsLauncher.launch_args._launch_args == {}


def test_set_quiet_launch():
    aprunLauncher = LaunchSettings(launcher=LauncherType.Alps)
    assert isinstance(aprunLauncher._arguments, AprunLaunchArguments)
    aprunLauncher.launch_args.set_quiet_launch(True)
    assert aprunLauncher.launch_args._launch_args == {"quiet": None}
    aprunLauncher.launch_args.set_quiet_launch(False)
    assert aprunLauncher.launch_args._launch_args == {}


def test_format_env_vars():
    env_vars = {"OMP_NUM_THREADS": "20", "LOGGING": "verbose"}
    aprunLauncher = LaunchSettings(launcher=LauncherType.Alps, env_vars=env_vars)
    assert isinstance(aprunLauncher._arguments, AprunLaunchArguments)
    aprunLauncher.update_env({"OMP_NUM_THREADS": "10"})
    formatted = aprunLauncher._arguments.format_env_vars(aprunLauncher._env_vars)
    result = ["-e", "OMP_NUM_THREADS=10", "-e", "LOGGING=verbose"]
    assert formatted == result


def test_aprun_settings():
    aprunLauncher = LaunchSettings(launcher=LauncherType.Alps)
    aprunLauncher.launch_args.set_cpus_per_task(2)
    aprunLauncher.launch_args.set_tasks(100)
    aprunLauncher.launch_args.set_tasks_per_node(20)
    formatted = aprunLauncher._arguments.format_launch_args()
    result = ["--cpus-per-pe=2", "--pes=100", "--pes-per-node=20"]
    assert formatted == result


def test_invalid_hostlist_format():
    """Test invalid hostlist formats"""
    alpsLauncher = LaunchSettings(launcher=LauncherType.Alps)
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_hostlist(["test", 5])
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_hostlist([5])
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_hostlist(5)


def test_invalid_exclude_hostlist_format():
    """Test invalid hostlist formats"""
    alpsLauncher = LaunchSettings(launcher=LauncherType.Alps)
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_excluded_hosts(["test", 5])
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_excluded_hosts([5])
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_excluded_hosts(5)


@pytest.mark.parametrize(
    "args, expected",
    (
        pytest.param({}, ("aprun", "--", "echo", "hello", "world"), id="Empty Args"),
        pytest.param(
            {"N": "1"},
            ("aprun", "-N", "1", "--", "echo", "hello", "world"),
            id="Short Arg",
        ),
        pytest.param(
            {"cpus-per-pe": "1"},
            ("aprun", "--cpus-per-pe=1", "--", "echo", "hello", "world"),
            id="Long Arg",
        ),
        pytest.param(
            {"q": None},
            ("aprun", "-q", "--", "echo", "hello", "world"),
            id="Short Arg (No Value)",
        ),
        pytest.param(
            {"quiet": None},
            ("aprun", "--quiet", "--", "echo", "hello", "world"),
            id="Long Arg (No Value)",
        ),
        pytest.param(
            {"N": "1", "cpus-per-pe": "123"},
            ("aprun", "-N", "1", "--cpus-per-pe=123", "--", "echo", "hello", "world"),
            id="Short and Long Args",
        ),
    ),
)
def test_formatting_launch_args(args, expected, test_dir):
    out = os.path.join(test_dir, "out.txt")
    err = os.path.join(test_dir, "err.txt")
    open(out, "w"), open(err, "w")
    shell_launch_cmd = _as_aprun_command(
        AprunLaunchArguments(args), ("echo", "hello", "world"), test_dir, {}, out, err
    )
    assert isinstance(shell_launch_cmd, ShellLauncherCommand)
    assert shell_launch_cmd.command_tuple == expected
    assert shell_launch_cmd.path == pathlib.Path(test_dir)
    assert shell_launch_cmd.env == {}
    assert isinstance(shell_launch_cmd.stdout, io.TextIOWrapper)
    assert shell_launch_cmd.stdout.name == out
    assert isinstance(shell_launch_cmd.stderr, io.TextIOWrapper)
    assert shell_launch_cmd.stderr.name == err
