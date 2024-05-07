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
import os
import os.path as osp
import stat
import sys

import pytest

from smartsim.error import LauncherError, SSUnsupportedError
from smartsim.settings.mpiSettings import (
    MpiexecSettings,
    MpirunSettings,
    OrterunSettings,
    _BaseMPISettings,
)

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


# Throw a warning instead of failing on machines without an MPI implementation
default_mpi_args = (sys.executable,)
default_mpi_kwargs = {"fail_if_missing_exec": False}


@pytest.mark.parametrize(
    "MPISettings", [MpirunSettings, MpiexecSettings, OrterunSettings]
)
def test_not_instanced_if_not_found(MPISettings):
    old_path = os.getenv("PATH")
    try:
        os.environ["PATH"] = ""
        with pytest.raises(LauncherError):
            MPISettings(*default_mpi_args)
    finally:
        os.environ["PATH"] = old_path


@pytest.mark.parametrize(
    "MPISettings,stubs_path,stub_exe",
    [
        pytest.param(
            MpirunSettings,
            osp.join("mpi_impl_stubs", "openmpi4"),
            "mpirun",
            id="OpenMPI4-mpirun",
        ),
        pytest.param(
            MpiexecSettings,
            osp.join("mpi_impl_stubs", "openmpi4"),
            "mpiexec",
            id="OpenMPI4-mpiexec",
        ),
        pytest.param(
            OrterunSettings,
            osp.join("mpi_impl_stubs", "openmpi4"),
            "orterun",
            id="OpenMPI4-orterun",
        ),
    ],
)
def test_expected_openmpi_instance_without_warning(
    MPISettings, stubs_path, stub_exe, fileutils, caplog
):
    from smartsim.settings.mpiSettings import logger

    old_path = os.environ.get("PATH")
    old_prop = logger.propagate
    logger.propagate = True

    try:
        stubs_path = fileutils.get_test_dir_path(stubs_path)
        stub_exe = osp.join(stubs_path, stub_exe)
        st = os.stat(stub_exe)
        if not st.st_mode & stat.S_IEXEC:
            os.chmod(stub_exe, st.st_mode | stat.S_IEXEC)

        os.environ["PATH"] = stubs_path
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            MPISettings(*default_mpi_args, **default_mpi_kwargs)
            for rec in caplog.records:
                if logging.WARNING <= rec.levelno:
                    pytest.fail(
                        (
                            "Unexepected log message when instancing valid "
                            "OpenMPI settings"
                        )
                    )
    finally:
        os.environ["PATH"] = old_path
        logger.propagate = old_prop


def test_error_if_slurm_mpiexec(fileutils):
    stubs_path = osp.join("mpi_impl_stubs", "slurm")
    stubs_path = fileutils.get_test_dir_path(stubs_path)
    stub_exe = osp.join(stubs_path, "mpiexec")
    old_path = os.environ.get("PATH")

    try:
        st = os.stat(stub_exe)
        if not st.st_mode & stat.S_IEXEC:
            os.chmod(stub_exe, st.st_mode | stat.S_IEXEC)

        os.environ["PATH"] = stubs_path
        with pytest.raises(SSUnsupportedError):
            MpiexecSettings(sys.executable)
    finally:
        os.environ["PATH"] = old_path


def test_base_settings():
    settings = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    settings.set_cpus_per_task(1)
    settings.set_tasks(2)
    settings.set_hostlist(["node005", "node006"])
    formatted = settings.format_run_args()
    result = ["--cpus-per-proc", "1", "--n", "2", "--host", "node005,node006"]
    assert formatted == result


def test_mpi_base_args():
    run_args = {
        "map-by": "ppr:1:node",
        "np": 1,
    }
    settings = _BaseMPISettings(
        *default_mpi_args, run_args=run_args, **default_mpi_kwargs
    )
    formatted = settings.format_run_args()
    result = ["--map-by", "ppr:1:node", "--np", "1"]
    assert formatted == result
    settings.set_task_map("ppr:2:node")
    formatted = settings.format_run_args()
    result = ["--map-by", "ppr:2:node", "--np", "1"]
    assert formatted == result


def test_mpi_add_mpmd():
    settings = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    settings_2 = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    settings.make_mpmd(settings_2)
    assert len(settings.mpmd) > 0
    assert settings.mpmd[0] == settings_2


def test_catch_colo_mpmd():
    settings = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    settings.colocated_fs_settings = {"port": 6379, "cpus": 1}
    settings_2 = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    with pytest.raises(SSUnsupportedError):
        settings.make_mpmd(settings_2)


def test_format_env():
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose"}
    settings = _BaseMPISettings(
        *default_mpi_args, env_vars=env_vars, **default_mpi_kwargs
    )
    settings.update_env({"OMP_NUM_THREADS": 10})
    formatted = settings.format_env_vars()
    result = [
        "-x",
        "OMP_NUM_THREADS=10",
        "-x",
        "LOGGING=verbose",
    ]
    assert formatted == result


def test_mpirun_hostlist_errors():
    settings = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    with pytest.raises(TypeError):
        settings.set_hostlist(4)


def test_mpirun_hostlist_errors_1():
    settings = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    with pytest.raises(TypeError):
        settings.set_hostlist([444])


@pytest.mark.parametrize("reserved_arg", ["wd", "wdir"])
def test_no_set_reserved_args(reserved_arg):
    srun = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    srun.set(reserved_arg)
    assert reserved_arg not in srun.run_args


def test_set_cpus_per_task():
    rs = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    rs.set_cpus_per_task(6)
    assert rs.run_args["cpus-per-proc"] == 6

    with pytest.raises(ValueError):
        rs.set_cpus_per_task("not an int")


def test_set_tasks_per_node():
    rs = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    rs.set_tasks_per_node(6)
    assert rs.run_args["npernode"] == 6

    with pytest.raises(ValueError):
        rs.set_tasks_per_node("not an int")


def test_set_tasks():
    rs = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    rs.set_tasks(6)
    assert rs.run_args["n"] == 6

    with pytest.raises(ValueError):
        rs.set_tasks("not an int")


def test_set_hostlist():
    rs = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    rs.set_hostlist(["host_A", "host_B"])
    assert rs.run_args["host"] == "host_A,host_B"

    rs.set_hostlist("host_A")
    assert rs.run_args["host"] == "host_A"

    with pytest.raises(TypeError):
        rs.set_hostlist([5])


def test_set_hostlist_from_file():
    rs = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    rs.set_hostlist_from_file("./path/to/hostfile")
    assert rs.run_args["hostfile"] == "./path/to/hostfile"

    rs.set_hostlist_from_file("~/other/file")
    assert rs.run_args["hostfile"] == "~/other/file"


def test_set_verbose():
    rs = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    rs.set_verbose_launch(True)
    assert "verbose" in rs.run_args

    rs.set_verbose_launch(False)
    assert "verbose" not in rs.run_args

    # Ensure not error on repeat calls
    rs.set_verbose_launch(False)


def test_quiet_launch():
    rs = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    rs.set_quiet_launch(True)
    assert "quiet" in rs.run_args

    rs.set_quiet_launch(False)
    assert "quiet" not in rs.run_args

    # Ensure not error on repeat calls
    rs.set_quiet_launch(False)


def test_set_broadcast():
    rs = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    rs.set_broadcast()
    assert "preload-binary" in rs.run_args

    rs.set_broadcast("/tmp/some/path")
    assert rs.run_args["preload-binary"] == None


def test_set_time():
    rs = _BaseMPISettings(*default_mpi_args, **default_mpi_kwargs)
    rs.set_time(minutes=1, seconds=12)
    assert rs.run_args["timeout"] == "72"

    rs.set_time(seconds=0)
    assert rs.run_args["timeout"] == "0"

    with pytest.raises(ValueError):
        rs.set_time("not an int")
