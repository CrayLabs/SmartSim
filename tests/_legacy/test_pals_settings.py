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


import os.path as osp
import shutil
import sys

import pytest

import smartsim._core.config.config
from smartsim._core.launcher import PBSLauncher
from smartsim._core.launcher.step.mpi_step import MpiexecStep
from smartsim.error import SSUnsupportedError
from smartsim.settings import PalsMpiexecSettings

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


default_exe = sys.executable
default_kwargs = {"fail_if_missing_exec": False}


@pytest.fixture(autouse=True)
def turn_off_telemetry_indirect(monkeypatch):
    monkeypatch.setattr(smartsim._core.config.config.Config, "telemetry_enabled", False)
    yield


# Uncomment when
# @pytest.mark.parametrize(
#    "function_name",[
#        'set_task_map',
#        'set_cpus_per_task',
#        'set_quiet_launch',
#        'set_walltime'
#    ]
# )
# def test_unsupported_methods(function_name):
#    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
#    func = getattr(settings, function_name)
#    with pytest.raises(SSUnsupportedError):
#        func(None)


def test_affinity_script():
    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
    settings.set_gpu_affinity_script("/path/to/set_affinity_gpu.sh", 1, 2)
    assert settings.format_run_args() == ["/path/to/set_affinity_gpu.sh", "1", "2"]


def test_cpu_binding_type():
    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
    settings.set_cpu_binding_type("numa")
    assert settings.format_run_args() == ["--cpu-bind", "numa"]


def test_tasks_per_node():
    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
    settings.set_tasks_per_node(48)
    assert settings.format_run_args() == ["--ppn", "48"]


def test_broadcast():
    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
    settings.set_broadcast()
    assert settings.format_run_args() == ["--transfer"]


def test_format_env_vars():
    example_env_vars = {"FOO_VERSION": "3.14", "PATH": None, "LD_LIBRARY_PATH": None}
    settings = PalsMpiexecSettings(
        default_exe, **default_kwargs, env_vars=example_env_vars
    )
    formatted = " ".join(settings.format_env_vars())
    expected = "--env FOO_VERSION=3.14 --envlist PATH,LD_LIBRARY_PATH"
    assert formatted == expected


@pytest.fixture
def mock_mpiexec(monkeypatch, fileutils):
    stub_path = fileutils.get_test_dir_path(osp.join("mpi_impl_stubs", "pals"))
    monkeypatch.setenv("PATH", stub_path, prepend=":")
    yield osp.join(stub_path, "mpiexec")


def set_env_var_to_inherit(rs):
    rs.env_vars["SPAM"] = None


@pytest.mark.parametrize(
    "rs_mutation, run_args",
    [
        pytest.param(
            lambda rs: rs.set_tasks(3),
            ["--np", "3"],
            id="set run args",
        ),
        pytest.param(
            set_env_var_to_inherit,
            ["--envlist", "SPAM"],
            id="env var [inherit from env]",
        ),
        pytest.param(
            lambda rs: rs.update_env({"SPAM": "EGGS"}),
            ["--env", "SPAM=EGGS"],
            id="env var [w/ val]",
        ),
    ],
)
def test_pbs_can_make_step_from_pals_settings_fmt_cmd(
    monkeypatch, mock_mpiexec, test_dir, rs_mutation, run_args
):
    # Setup run settings
    exe_args = ["-c", """'print("Hello")'"""]
    rs = PalsMpiexecSettings(sys.executable, exe_args)
    rs_mutation(rs)

    # setup a launcher and pretend we are in an alloc
    launcher = PBSLauncher()
    monkeypatch.setenv(f"PBS_JOBID", "mock-job")

    wdir = test_dir
    step = launcher.create_step("my_step", wdir, rs)
    assert isinstance(step, MpiexecStep)
    assert step.get_launch_cmd() == [
        mock_mpiexec,
        "--wdir",
        wdir,
        *run_args,
        sys.executable,
        *exe_args,
    ]


def test_pals_settings_can_be_correctly_made_mpmd(monkeypatch, test_dir, mock_mpiexec):
    # Setup run settings
    def make_rs(exe, exe_args):
        return PalsMpiexecSettings(exe, exe_args), [exe] + exe_args

    echo = shutil.which("echo")
    rs_1, expected_exe_1 = make_rs(echo, ["spam"])
    rs_2, expected_exe_2 = make_rs(echo, ["and"])
    rs_3, expected_exe_3 = make_rs(echo, ["eggs"])

    # modify run args
    def set_tasks(rs, num):
        rs.set_tasks(num)
        return rs, ["--np", str(num)]

    rs_1, expected_rs_1 = set_tasks(rs_1, 5)
    rs_2, expected_rs_2 = set_tasks(rs_2, 2)
    rs_3, expected_rs_3 = set_tasks(rs_3, 8)

    # MPMD it up
    rs_1.make_mpmd(rs_2)
    rs_1.make_mpmd(rs_3)

    # setup a launcher and pretend we are in an alloc
    launcher = PBSLauncher()
    monkeypatch.setenv(f"PBS_JOBID", "mock-job")

    wdir = test_dir
    step = launcher.create_step("my_step", wdir, rs_1)
    assert isinstance(step, MpiexecStep)
    assert step.get_launch_cmd() == [
        mock_mpiexec,
        "--wdir",
        wdir,
        # rs 1
        *expected_rs_1,
        *expected_exe_1,
        ":",
        # rs 2
        *expected_rs_2,
        *expected_exe_2,
        ":",
        # rs 3
        *expected_rs_3,
        *expected_exe_3,
    ]
