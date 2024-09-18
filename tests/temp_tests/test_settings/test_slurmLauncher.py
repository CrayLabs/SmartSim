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

from smartsim._core.shell.shell_launcher import ShellLauncherCommand
from smartsim.settings import LaunchSettings
from smartsim.settings.arguments.launch.slurm import (
    SlurmLaunchArguments,
    _as_srun_command,
)
from smartsim.settings.launch_command import LauncherType

pytestmark = pytest.mark.group_a


def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    ls = LaunchSettings(launcher=LauncherType.Slurm)
    assert ls.launch_args.launcher_str() == LauncherType.Slurm.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,), "2", "nodes", id="set_nodes"),
        pytest.param(
            "set_hostlist", ("host_A",), "host_A", "nodelist", id="set_hostlist_str"
        ),
        pytest.param(
            "set_hostlist",
            (["host_A", "host_B"],),
            "host_A,host_B",
            "nodelist",
            id="set_hostlist_list[str]",
        ),
        pytest.param(
            "set_hostlist_from_file",
            ("./path/to/hostfile",),
            "./path/to/hostfile",
            "nodefile",
            id="set_hostlist_from_file",
        ),
        pytest.param(
            "set_excluded_hosts",
            ("host_A",),
            "host_A",
            "exclude",
            id="set_excluded_hosts_str",
        ),
        pytest.param(
            "set_excluded_hosts",
            (["host_A", "host_B"],),
            "host_A,host_B",
            "exclude",
            id="set_excluded_hosts_list[str]",
        ),
        pytest.param(
            "set_cpus_per_task", (4,), "4", "cpus-per-task", id="set_cpus_per_task"
        ),
        pytest.param("set_tasks", (4,), "4", "ntasks", id="set_tasks"),
        pytest.param(
            "set_tasks_per_node", (4,), "4", "ntasks-per-node", id="set_tasks_per_node"
        ),
        pytest.param(
            "set_cpu_bindings", (4,), "map_cpu:4", "cpu_bind", id="set_cpu_bindings"
        ),
        pytest.param(
            "set_cpu_bindings",
            ([4, 4],),
            "map_cpu:4,4",
            "cpu_bind",
            id="set_cpu_bindings_list[str]",
        ),
        pytest.param(
            "set_memory_per_node", (8000,), "8000M", "mem", id="set_memory_per_node"
        ),
        pytest.param(
            "set_executable_broadcast",
            ("/tmp/some/path",),
            "/tmp/some/path",
            "bcast",
            id="set_broadcast",
        ),
        pytest.param("set_node_feature", ("P100",), "P100", "C", id="set_node_feature"),
        pytest.param(
            "set_walltime", ("10:00:00",), "10:00:00", "time", id="set_walltime"
        ),
    ],
)
def test_slurm_class_methods(function, value, flag, result):
    slurmLauncher = LaunchSettings(launcher=LauncherType.Slurm)
    assert isinstance(slurmLauncher.launch_args, SlurmLaunchArguments)
    getattr(slurmLauncher.launch_args, function)(*value)
    assert slurmLauncher.launch_args._launch_args[flag] == result


def test_set_verbose_launch():
    ls = LaunchSettings(launcher=LauncherType.Slurm)
    ls.launch_args.set_verbose_launch(True)
    assert ls.launch_args._launch_args == {"verbose": None}
    ls.launch_args.set_verbose_launch(False)
    assert ls.launch_args._launch_args == {}


def test_set_quiet_launch():
    ls = LaunchSettings(launcher=LauncherType.Slurm)
    ls.launch_args.set_quiet_launch(True)
    assert ls.launch_args._launch_args == {"quiet": None}
    ls.launch_args.set_quiet_launch(False)
    assert ls.launch_args._launch_args == {}


def test_format_env_vars():
    """Test format_env_vars runs correctly"""
    env_vars = {
        "OMP_NUM_THREADS": "20",
        "LOGGING": "verbose",
        "SSKEYIN": "name_0,name_1",
    }
    ls = LaunchSettings(launcher=LauncherType.Slurm, env_vars=env_vars)
    ls_format = ls._arguments.format_env_vars(env_vars)
    assert "OMP_NUM_THREADS=20" in ls_format
    assert "LOGGING=verbose" in ls_format
    assert all("SSKEYIN" not in x for x in ls_format)


def test_catch_existing_env_var(caplog, monkeypatch):
    slurmSettings = LaunchSettings(
        launcher=LauncherType.Slurm,
        env_vars={
            "SMARTSIM_TEST_VAR": "B",
        },
    )
    monkeypatch.setenv("SMARTSIM_TEST_VAR", "A")
    monkeypatch.setenv("SMARTSIM_TEST_CSVAR", "A,B")
    caplog.clear()
    slurmSettings._arguments.format_env_vars(slurmSettings._env_vars)

    msg = f"Variable SMARTSIM_TEST_VAR is set to A in current environment. "
    msg += f"If the job is running in an interactive allocation, the value B will not be set. "
    msg += "Please consider removing the variable from the environment and re-running the experiment."

    for record in caplog.records:
        assert record.levelname == "WARNING"
        assert record.message == msg

    caplog.clear()

    env_vars = {"SMARTSIM_TEST_VAR": "B", "SMARTSIM_TEST_CSVAR": "C,D"}
    settings = LaunchSettings(launcher=LauncherType.Slurm, env_vars=env_vars)
    settings._arguments.format_comma_sep_env_vars(env_vars)

    for record in caplog.records:
        assert record.levelname == "WARNING"
        assert record.message == msg


def test_format_comma_sep_env_vars():
    """Test format_comma_sep_env_vars runs correctly"""
    env_vars = {
        "OMP_NUM_THREADS": "20",
        "LOGGING": "verbose",
        "SSKEYIN": "name_0,name_1",
    }
    slurmLauncher = LaunchSettings(launcher=LauncherType.Slurm, env_vars=env_vars)
    formatted, comma_separated_formatted = (
        slurmLauncher._arguments.format_comma_sep_env_vars(env_vars)
    )
    assert "OMP_NUM_THREADS" in formatted
    assert "LOGGING" in formatted
    assert "SSKEYIN" in formatted
    assert "name_0,name_1" not in formatted
    assert "SSKEYIN=name_0,name_1" in comma_separated_formatted


def test_slurmSettings_settings():
    """Test format_launch_args runs correctly"""
    slurmLauncher = LaunchSettings(launcher=LauncherType.Slurm)
    slurmLauncher.launch_args.set_nodes(5)
    slurmLauncher.launch_args.set_cpus_per_task(2)
    slurmLauncher.launch_args.set_tasks(100)
    slurmLauncher.launch_args.set_tasks_per_node(20)
    formatted = slurmLauncher._arguments.format_launch_args()
    result = ["--nodes=5", "--cpus-per-task=2", "--ntasks=100", "--ntasks-per-node=20"]
    assert formatted == result


def test_slurmSettings_launch_args():
    """Test the possible user overrides through run_args"""
    launch_args = {
        "account": "A3123",
        "exclusive": None,
        "C": "P100",  # test single letter variables
        "nodes": 10,
        "ntasks": 100,
    }
    slurmLauncher = LaunchSettings(launcher=LauncherType.Slurm, launch_args=launch_args)
    formatted = slurmLauncher._arguments.format_launch_args()
    result = [
        "--account=A3123",
        "--exclusive",
        "-C",
        "P100",
        "--nodes=10",
        "--ntasks=100",
    ]
    assert formatted == result


def test_invalid_hostlist_format():
    """Test invalid hostlist formats"""
    slurmLauncher = LaunchSettings(launcher=LauncherType.Slurm)
    with pytest.raises(TypeError):
        slurmLauncher.launch_args.set_hostlist(["test", 5])
    with pytest.raises(TypeError):
        slurmLauncher.launch_args.set_hostlist([5])
    with pytest.raises(TypeError):
        slurmLauncher.launch_args.set_hostlist(5)


def test_invalid_exclude_hostlist_format():
    """Test invalid hostlist formats"""
    slurmLauncher = LaunchSettings(launcher=LauncherType.Slurm)
    with pytest.raises(TypeError):
        slurmLauncher.launch_args.set_excluded_hosts(["test", 5])
    with pytest.raises(TypeError):
        slurmLauncher.launch_args.set_excluded_hosts([5])
    with pytest.raises(TypeError):
        slurmLauncher.launch_args.set_excluded_hosts(5)


def test_invalid_node_feature_format():
    """Test invalid node feature formats"""
    slurmLauncher = LaunchSettings(launcher=LauncherType.Slurm)
    with pytest.raises(TypeError):
        slurmLauncher.launch_args.set_node_feature(["test", 5])
    with pytest.raises(TypeError):
        slurmLauncher.launch_args.set_node_feature([5])
    with pytest.raises(TypeError):
        slurmLauncher.launch_args.set_node_feature(5)


def test_invalid_walltime_format():
    """Test invalid walltime formats"""
    slurmLauncher = LaunchSettings(launcher=LauncherType.Slurm)
    with pytest.raises(ValueError):
        slurmLauncher.launch_args.set_walltime("11:11")
    with pytest.raises(ValueError):
        slurmLauncher.launch_args.set_walltime("ss:ss:ss")
    with pytest.raises(ValueError):
        slurmLauncher.launch_args.set_walltime("11:ss:ss")
    with pytest.raises(ValueError):
        slurmLauncher.launch_args.set_walltime("0s:ss:ss")


def test_set_het_groups(monkeypatch):
    """Test ability to set one or more het groups to run setting"""
    monkeypatch.setenv("SLURM_HET_SIZE", "4")
    slurmLauncher = LaunchSettings(launcher=LauncherType.Slurm)
    slurmLauncher.launch_args.set_het_group([1])
    assert slurmLauncher._arguments._launch_args["het-group"] == "1"
    slurmLauncher.launch_args.set_het_group([3, 2])
    assert slurmLauncher._arguments._launch_args["het-group"] == "3,2"
    with pytest.raises(ValueError):
        slurmLauncher.launch_args.set_het_group([4])


@pytest.mark.parametrize(
    "args, expected",
    (
        pytest.param(
            {},
            (
                "srun",
                "--output=output.txt",
                "--error=error.txt",
                "--",
                "echo",
                "hello",
                "world",
            ),
            id="Empty Args",
        ),
        pytest.param(
            {"N": "1"},
            (
                "srun",
                "-N",
                "1",
                "--output=output.txt",
                "--error=error.txt",
                "--",
                "echo",
                "hello",
                "world",
            ),
            id="Short Arg",
        ),
        pytest.param(
            {"nodes": "1"},
            (
                "srun",
                "--nodes=1",
                "--output=output.txt",
                "--error=error.txt",
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
                "srun",
                "-v",
                "--output=output.txt",
                "--error=error.txt",
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
                "srun",
                "--verbose",
                "--output=output.txt",
                "--error=error.txt",
                "--",
                "echo",
                "hello",
                "world",
            ),
            id="Long Arg (No Value)",
        ),
        pytest.param(
            {"nodes": "1", "n": "123"},
            (
                "srun",
                "--nodes=1",
                "-n",
                "123",
                "--output=output.txt",
                "--error=error.txt",
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
    shell_launch_cmd = _as_srun_command(
        args=SlurmLaunchArguments(args),
        exe=("echo", "hello", "world"),
        path=test_dir,
        env={},
        stdout_path="output.txt",
        stderr_path="error.txt",
    )
    assert isinstance(shell_launch_cmd, ShellLauncherCommand)
    assert shell_launch_cmd.command_tuple == expected
    assert shell_launch_cmd.path == test_dir
    assert shell_launch_cmd.env == {}
    assert shell_launch_cmd.stdout == subprocess.DEVNULL
    assert shell_launch_cmd.stderr == subprocess.DEVNULL
