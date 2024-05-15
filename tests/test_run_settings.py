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


import itertools
import logging
import os.path as osp

import pytest

from smartsim.error.errors import SSUnsupportedError
from smartsim.settings import (
    MpiexecSettings,
    MpirunSettings,
    OrterunSettings,
    PalsMpiexecSettings,
    RunSettings,
    Singularity,
)
from smartsim.settings.settings import create_run_settings

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.slow_tests


def test_create_run_settings_local():
    # no run command provided
    rs = create_run_settings("local", "echo", "hello", run_command=None)
    assert rs.run_command == None
    assert type(rs) == RunSettings

    # auto should never return a run_command when
    # the user has specified the local launcher
    auto = create_run_settings("local", "echo", "hello", run_command="auto")
    assert auto.run_command == None
    assert type(auto) == RunSettings

    # Test when a run_command is provided that we do not currently have a helper
    # implementation for it.
    # NOTE: we allow for the command to be invalid if it's user specified in the
    # case where a head node may not have the same installed binaries as the MOM
    # or compute nodes.
    specific = create_run_settings("local", "echo", "hello", run_command="specific")
    assert specific.run_command == "specific"
    assert type(specific) == RunSettings


@pytest.mark.parametrize(
    "launcher, run_cmd, stub_path, type_mpi_settings",
    [
        # Use OpenMPI style settigs for all launchers
        *itertools.chain.from_iterable(
            (
                (
                    pytest.param(
                        l,
                        "mpirun",
                        osp.join("mpi_impl_stubs", "openmpi4"),
                        MpirunSettings,
                        id=f"{l}/mpirun",
                    ),
                    pytest.param(
                        l,
                        "mpiexec",
                        osp.join("mpi_impl_stubs", "openmpi4"),
                        MpiexecSettings,
                        id=f"{l}/mpiexec",
                    ),
                    pytest.param(
                        l,
                        "orterun",
                        osp.join("mpi_impl_stubs", "openmpi4"),
                        OrterunSettings,
                        id=f"{l}/orterun",
                    ),
                )
                for l in ("local", "pbs", "slurm", "lsf")
            )
        ),
        # Except for launchers that implement their own MPI settings
        pytest.param(
            "pals",
            "mpiexec",
            osp.join("mpi_impl_stubs", "pals"),
            PalsMpiexecSettings,
            id="pals/mpiexec",
        ),
    ],
)
def test_create_run_settings_returns_expected_settings_subclass_for_mpi_variants(
    monkeypatch, fileutils, launcher, run_cmd, stub_path, type_mpi_settings
):
    stub_path = fileutils.get_test_dir_path(stub_path)
    monkeypatch.setenv("PATH", stub_path, prepend=":")
    settings = create_run_settings(launcher, "echo", "hello", run_command=run_cmd)
    assert settings.run_command == osp.join(stub_path, run_cmd)
    assert isinstance(settings, type_mpi_settings)


def test_create_run_settings_raise_if_slurm_mpiexec_wrapper_detected(
    monkeypatch, fileutils
):
    monkeypatch.setenv(
        "PATH",
        fileutils.get_test_dir_path(osp.join("mpi_impl_stubs", "slurm")),
        prepend=":",
    )
    with pytest.raises(SSUnsupportedError):
        create_run_settings("slurm", "echo", ["hello", "world"], run_command="mpiexec")


def test_create_run_settings_input_mutation():
    # Tests that the run args passed in are not modified after initialization
    key0, key1, key2 = "arg0", "arg1", "arg2"
    val0, val1, val2 = "val0", "val1", "val2"

    default_run_args = {
        key0: val0,
        key1: val1,
        key2: val2,
    }
    rs0 = create_run_settings(
        "local", "echo", "hello", run_command="auto", run_args=default_run_args
    )

    # Confirm initial values are set
    assert rs0.run_args[key0] == val0
    assert rs0.run_args[key1] == val1
    assert rs0.run_args[key2] == val2

    # Update our common run arguments
    val2_upd = f"not-{val2}"
    default_run_args[key2] = val2_upd

    # Confirm previously created run settings are not changed
    assert rs0.run_args[key2] == val2


####### Base Run Settings tests #######


def test_add_exe_args():
    """Ensure that valid exe args are added correctly"""
    settings = RunSettings("python")
    settings.add_exe_args("--time 5")
    settings.add_exe_args(["--add", "--list"])
    result = ["--time", "5", "--add", "--list"]
    assert settings.exe_args == result


def test_add_exe_args_list_of_ints():
    """Ensure that non-string exe args fail validation"""
    settings = RunSettings("python")
    with pytest.raises(TypeError):
        settings.add_exe_args([1, 2, 3])


def test_add_exe_args_list_of_mixed():
    """Ensure that any non-string exe arg fails validation for all"""
    settings = RunSettings("python")
    with pytest.raises(TypeError):
        settings.add_exe_args(["1", "2", 3])


def test_add_exe_args_list_of_lists():
    """Ensure that any non-string exe arg fails validation for all"""
    settings = RunSettings("python")
    with pytest.raises(TypeError):
        settings.add_exe_args(["1", "2", "3"], ["1", "2", "3"])


def test_init_exe_args_list_of_lists():
    """Ensure that a list of lists exe arg fails validation"""
    exe_args = [["1", "2", "3"], ["4", "5", "6"]]
    with pytest.raises(TypeError):
        _ = RunSettings("python", exe_args=exe_args)


def test_init_exe_args_list_of_lists_mixed():
    """Ensure that a list of lists exe arg fails validation"""
    exe_args = [["1", "2", 3], ["4", "5", 6]]
    with pytest.raises(TypeError):
        _ = RunSettings("python", exe_args=exe_args)


def test_add_exe_args_space_delimited_string():
    """Ensure that any non-string exe arg fails validation for all"""
    settings = RunSettings("python")
    expected = ["1", "2", "3"]
    settings.add_exe_args("1 2 3")

    assert settings.exe_args == expected


def test_format_run_args():
    settings = RunSettings(
        "echo", exe_args="test", run_command="mpirun", run_args={"-np": 2}
    )
    run_args = settings.format_run_args()
    assert type(run_args) == type(list())
    assert run_args == ["-np", "2"]


def test_addto_existing_exe_args():
    list_exe_args_settings = RunSettings("python", ["sleep.py", "--time=5"])
    str_exe_args_settings = RunSettings("python", "sleep.py --time=5")

    # both should be the same
    args = ["sleep.py", "--time=5"]
    assert list_exe_args_settings.exe_args == args
    assert str_exe_args_settings.exe_args == args

    # add to exe_args
    list_exe_args_settings.add_exe_args("--stop=10")
    str_exe_args_settings.add_exe_args(["--stop=10"])

    args = ["sleep.py", "--time=5", "--stop=10"]
    assert list_exe_args_settings.exe_args == args
    assert str_exe_args_settings.exe_args == args


def test_existing_exe_args_mutation():
    """
    Ensure that if the argument list is changed, any previously
    created run settings don't reflect the change due to pass-by-ref
    """
    args = ["sleep.py", "--time=5"]
    orig = ["sleep.py", "--time=5"]
    rs0 = RunSettings("python", args)

    # both should be the same
    assert rs0.exe_args == args

    # modify the args list
    args.append("--foo")
    assert rs0.exe_args == orig

    # create another run settings instance
    rs1 = RunSettings("python", args)
    assert rs1.exe_args == args
    assert rs0.exe_args != rs1.exe_args


def test_direct_set_exe_args_mutation():
    """
    Ensure that if the argument list is set directly, any previously
    created run settings don't reflect the change due to pass-by-ref
    """
    args = ["sleep.py", "--time=5"]
    orig = ["sleep.py", "--time=5"]
    rs0 = RunSettings("python")
    rs0.exe_args = args

    # both should be the same
    assert rs0.exe_args == args

    # modify the args list
    args.append("--foo")
    assert rs0.exe_args == orig

    # create another run settings instance
    rs1 = RunSettings("python")
    rs1.exe_args = args
    assert rs1.exe_args == args
    assert rs0.exe_args != rs1.exe_args


def test_bad_exe_args():
    """test when user provides incorrect types to exe_args"""
    exe_args = {"dict": "is-wrong-type"}
    with pytest.raises(TypeError):
        _ = RunSettings("python", exe_args=exe_args)


def test_bad_exe_args_2():
    """test when user provides incorrect types to exe_args"""
    exe_args = ["list-includes-int", 5]
    with pytest.raises(TypeError):
        _ = RunSettings("python", exe_args=exe_args)


def test_set_args():
    rs = RunSettings("python")
    rs.set("str", "some-string")
    rs.set("nothing")

    assert "str" in rs.run_args
    assert rs.run_args["str"] == "some-string"

    assert "nothing" in rs.run_args
    assert rs.run_args["nothing"] is None


@pytest.mark.parametrize(
    "set_str,val,key",
    [
        pytest.param("normal-key", "some-val", "normal-key", id="set string"),
        pytest.param("--a-key", "a-value", "a-key", id="strip doulbe dashes"),
        pytest.param("-b", "some-str", "b", id="strip single dashes"),
        pytest.param("   c    ", "some-val", "c", id="strip spaces"),
        pytest.param("   --a-mess    ", "5", "a-mess", id="strip everything"),
    ],
)
def test_set_format_args(set_str, val, key):
    rs = RunSettings("python")
    rs.set(set_str, val)
    assert rs.run_args[key] == val


@pytest.mark.parametrize(
    "method,params",
    [
        pytest.param("set_nodes", (2,), id="set_nodes"),
        pytest.param("set_tasks", (2,), id="set_tasks"),
        pytest.param("set_tasks_per_node", (3,), id="set_tasks_per_node"),
        pytest.param("set_task_map", (3,), id="set_task_map"),
        pytest.param("set_cpus_per_task", (4,), id="set_cpus_per_task"),
        pytest.param("set_hostlist", ("hostlist",), id="set_hostlist"),
        pytest.param("set_node_feature", ("P100",), id="set_node_feature"),
        pytest.param(
            "set_hostlist_from_file", ("~/hostfile",), id="set_hostlist_from_file"
        ),
        pytest.param("set_excluded_hosts", ("hostlist",), id="set_excluded_hosts"),
        pytest.param("set_cpu_bindings", ([1, 2, 3],), id="set_cpu_bindings"),
        pytest.param("set_memory_per_node", (16_000,), id="set_memory_per_node"),
        pytest.param("set_verbose_launch", (False,), id="set_verbose_launch"),
        pytest.param("set_quiet_launch", (True,), id="set_quiet_launch"),
        pytest.param("set_broadcast", ("/tmp",), id="set_broadcast"),
        pytest.param("set_time", (0, 0, 0), id="set_time"),
        pytest.param("set_walltime", ("00:55:00",), id="set_walltime"),
        pytest.param("set_binding", ("packed:21",), id="set_binding"),
        pytest.param("set_mpmd_preamble", (["list", "strs"],), id="set_mpmd_preamble"),
        pytest.param("make_mpmd", (None,), id="make_mpmd"),
    ],
)
def test_unimplimented_setters_throw_warning(caplog, method, params):
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        rs = RunSettings("python")
        try:
            getattr(rs, method)(*params)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and "not implemented" in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )


def test_base_format_env_vars():
    rs = RunSettings(
        "python",
        env_vars={
            "A": "a",
            "B": None,
            "C": "",
            "D": 12,
        },
    )
    assert rs.format_env_vars() == ["A=a", "B=", "C=", "D=12"]


def test_set_raises_type_errors():
    rs = RunSettings("python")

    with pytest.raises(TypeError):
        rs.set("good-key", 5)

    with pytest.raises(TypeError):
        rs.set(9)


def test_set_overwrites_prev_args():
    rs = RunSettings("python")
    rs.set("some-key", "some-val")
    rs.set("some-key", "another-val")
    assert rs.run_args["some-key"] == "another-val"


def test_set_conditional():
    rs = RunSettings("python")
    ans = 2 + 2
    rs.set("ans-is-4-arg", condition=ans == 4)
    rs.set("ans-is-5-arg", condition=ans == 5)
    assert "ans-is-4-arg" in rs.run_args
    assert "ans-is-5-arg" not in rs.run_args


def test_container_check():
    """Ensure path is expanded when run outside of a container"""
    sample_exe = "python"
    containerURI = "docker://alrigazzi/smartsim-testing:latest"
    container = Singularity(containerURI)

    rs = RunSettings(sample_exe, container=container)
    assert sample_exe in rs.exe

    rs = RunSettings(sample_exe, container=None)
    assert len(rs.exe[0]) > len(sample_exe)


def test_run_command():
    """Ensure that run_command expands cmd as needed"""
    sample_exe = "python"
    cmd = "echo"

    rs = RunSettings(sample_exe, run_command=cmd)
    rc_output: str = rs.run_command or ""
    assert len(rc_output) > len(cmd)


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
    sample_exe = "python"
    cmd = "echo"

    orig_env = {"key": "value"}
    rs = RunSettings(sample_exe, run_command=cmd, env_vars=orig_env)
    rs.update_env(env_vars)

    combined_keys = {k for k in env_vars.keys()}
    combined_keys.update(k for k in orig_env.keys())

    assert len(rs.env_vars) == len(combined_keys)
    assert {k for k in rs.env_vars.keys()} == combined_keys


def test_env_vars_mutation():
    """
    Ensure that if the env_vars dict is changed, any previously
    created run settings don't reflect the change due to pass-by-ref
    """
    sample_exe = "python"
    cmd = "echo"

    env_vars = {"k1": "v1", "k2": "v2"}
    orig_env = {"k1": "v1", "k2": "v2"}
    rs = RunSettings(sample_exe, run_command=cmd, env_vars=env_vars)

    # verify initial expectations
    assert len(rs.env_vars) == len(env_vars)
    assert rs.env_vars == orig_env

    # update a value in the env_vars dict & verify
    # that the run settings do not reflect the change
    env_vars["k1"] = f"not-{env_vars['k1']}"
    assert rs.env_vars["k1"] != env_vars["k1"]
    assert rs.env_vars["k1"] == orig_env["k1"]


def test_direct_set_env_vars_mutation():
    """
    Ensure that if the env_vars dict is explicitly set, any previously
    created run settings don't reflect the change due to pass-by-ref
    """
    sample_exe = "python"
    cmd = "echo"

    env_vars = {"k1": "v1", "k2": "v2"}
    orig_env = {"k1": "v1", "k2": "v2"}
    rs = RunSettings(sample_exe, run_command=cmd)
    rs.env_vars = env_vars

    # verify initial expectations
    assert len(rs.env_vars) == len(env_vars)
    assert rs.env_vars == orig_env

    # update a value in the env_vars dict & verify
    # that the run settings do not reflect the change
    env_vars["k1"] = f"not-{env_vars['k1']}"
    assert rs.env_vars["k1"] != env_vars["k1"]
    assert rs.env_vars["k1"] == orig_env["k1"]


@pytest.mark.parametrize(
    "env_vars",
    [
        pytest.param({}, id="no env vars"),
        pytest.param({"env1": "abc"}, id="normal var"),
        pytest.param({"env1": "abc,def"}, id="compound var"),
        pytest.param({"env1": "xyz", "env2": "pqr"}, id="multiple env vars"),
    ],
)
def test_update_env_empty(env_vars):
    """Ensure non-initialized env vars update correctly"""
    sample_exe = "python"
    cmd = "echo"

    rs = RunSettings(sample_exe, run_command=cmd)
    rs.update_env(env_vars)

    assert len(rs.env_vars) == len(env_vars.keys())


def test_update_env():
    """Ensure empty env vars is handled gracefully"""
    sample_exe = "python"
    cmd = "echo"

    rs = RunSettings(sample_exe, run_command=cmd)

    env_vars = {}
    assert not rs.env_vars


@pytest.mark.parametrize(
    "env_vars",
    [
        pytest.param({"env1": None}, id="null value not allowed"),
        pytest.param({"env1": {"abc"}}, id="set value not allowed"),
        pytest.param({"env1": {"abc": "def"}}, id="dict value not allowed"),
    ],
)
def test_update_env_null_valued(env_vars):
    """Ensure validation of env var in update"""
    sample_exe = "python"
    cmd = "echo"
    orig_env = {}

    with pytest.raises(TypeError) as ex:
        rs = RunSettings(sample_exe, run_command=cmd, env_vars=orig_env)
        rs.update_env(env_vars)
