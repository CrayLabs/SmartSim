import pytest

import os
import sys

from smartsim.settings.mpirunSettings import (
    _OpenMPISettings,
    MpirunSettings,
    MpiexecSettings,
    OrterunSettings,
)
from smartsim.error import SSUnsupportedError


@pytest.mark.parametrize(
    "OpenMPISettings", [MpirunSettings, MpiexecSettings, OrterunSettings]
)
def test_not_instanced_if_not_found(OpenMPISettings):
    old_path = os.getenv("PATH")
    try:
        os.environ["PATH"] = ""
        with pytest.raises(FileNotFoundError):
            OpenMPISettings(sys.executable)
    finally:
        os.environ["PATH"] = old_path


def test_openmpi_base_settings():
    settings = _OpenMPISettings("python")
    settings.set_cpus_per_task(1)
    settings.set_tasks(2)
    settings.set_hostlist(["node005", "node006"])
    formatted = settings.format_run_args()
    result = ["--cpus-per-proc", "1", "--n", "2", "--host", "node005,node006"]
    assert formatted == result


def test_openmpi_base_args():
    run_args = {
        "map-by": "ppr:1:node",
        "np": 1,
    }
    settings = _OpenMPISettings("python", run_args=run_args)
    formatted = settings.format_run_args()
    result = ["--map-by", "ppr:1:node", "--np", "1"]
    assert formatted == result
    settings.set_task_map("ppr:2:node")
    formatted = settings.format_run_args()
    result = ["--map-by", "ppr:2:node", "--np", "1"]
    assert formatted == result


def test_openmpi_add_mpmd():
    settings = _OpenMPISettings("python")
    settings_2 = _OpenMPISettings("python")
    settings.make_mpmd(settings_2)
    assert len(settings.mpmd) > 0
    assert settings.mpmd[0] == settings_2


def test_catch_colo_mpmd():
    settings = _OpenMPISettings("python")
    settings.colocated_db_settings = {"port": 6379, "cpus": 1}
    settings_2 = _OpenMPISettings("python")
    with pytest.raises(SSUnsupportedError):
        settings.make_mpmd(settings_2)


def test_format_env():
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose"}
    settings = _OpenMPISettings("python", env_vars=env_vars)
    settings.update_env({"OMP_NUM_THREADS": 10})
    formatted = settings.format_env_vars()
    result = [
        "-x",
        "PATH",
        "-x",
        "LD_LIBRARY_PATH",
        "-x",
        "PYTHONPATH",
        "-x",
        "OMP_NUM_THREADS=10",
        "-x",
        "LOGGING=verbose",
    ]
    assert formatted == result


def test_mpirun_hostlist_errors():
    settings = _OpenMPISettings("python")
    with pytest.raises(TypeError):
        settings.set_hostlist(4)


def test_mpirun_hostlist_errors_1():
    settings = _OpenMPISettings("python")
    with pytest.raises(TypeError):
        settings.set_hostlist([444])


@pytest.mark.parametrize("reserved_arg", ["wd", "wdir"])
def test_no_set_reserved_args(reserved_arg):
    srun = _OpenMPISettings("python")
    srun.set(reserved_arg)
    assert reserved_arg not in srun.run_args


def test_set_cpus_per_task():
    rs = _OpenMPISettings("python")
    rs.set_cpus_per_task(6)
    assert rs.run_args["cpus-per-proc"] == 6

    with pytest.raises(ValueError):
        rs.set_cpus_per_task("not an int")


def test_set_tasks_per_node():
    rs = _OpenMPISettings("python")
    rs.set_tasks_per_node(6)
    assert rs.run_args["npernode"] == 6

    with pytest.raises(ValueError):
        rs.set_tasks_per_node("not an int")


def test_set_tasks():
    rs = _OpenMPISettings("python")
    rs.set_tasks(6)
    assert rs.run_args["n"] == 6

    with pytest.raises(ValueError):
        rs.set_tasks("not an int")


def test_set_hostlist():
    rs = _OpenMPISettings("python")
    rs.set_hostlist(["host_A", "host_B"])
    assert rs.run_args["host"] == "host_A,host_B"

    rs.set_hostlist("host_A")
    assert rs.run_args["host"] == "host_A"

    with pytest.raises(TypeError):
        rs.set_hostlist([5])


def test_set_hostlist_from_file():
    rs = _OpenMPISettings("python")
    rs.set_hostlist_from_file("./path/to/hostfile")
    assert rs.run_args["hostfile"] == "./path/to/hostfile"

    rs.set_hostlist_from_file("~/other/file")
    assert rs.run_args["hostfile"] == "~/other/file"


def test_set_verbose():
    rs = _OpenMPISettings("python")
    rs.set_verbose_launch(True)
    assert "verbose" in rs.run_args

    rs.set_verbose_launch(False)
    assert "verbose" not in rs.run_args

    # Ensure not error on repeat calls
    rs.set_verbose_launch(False)


def test_quiet_launch():
    rs = _OpenMPISettings("python")
    rs.set_quiet_launch(True)
    assert "quiet" in rs.run_args

    rs.set_quiet_launch(False)
    assert "quiet" not in rs.run_args

    # Ensure not error on repeat calls
    rs.set_quiet_launch(False)


def test_set_broadcast():
    rs = _OpenMPISettings("python")
    rs.set_broadcast()
    assert "preload-binary" in rs.run_args

    rs.set_broadcast("/tmp/some/path")
    assert rs.run_args["preload-binary"] == None


def test_set_time():
    rs = _OpenMPISettings("python")
    rs.set_time(minutes=1, seconds=12)
    assert rs.run_args["timeout"] == "72"

    rs.set_time(seconds=0)
    assert rs.run_args["timeout"] == "0"

    with pytest.raises(ValueError):
        rs.set_time("not an int")
