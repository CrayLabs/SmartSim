import pytest

from smartsim.settings import MpirunSettings
from smartsim.error import SSUnsupportedError

def test_mpirun_settings():
    settings = MpirunSettings("python")
    settings.set_cpus_per_task(1)
    settings.set_tasks(2)
    settings.set_hostlist(["node005", "node006"])
    formatted = settings.format_run_args()
    result = ["--cpus-per-proc", "1", "--n", "2", "--host", "node005,node006"]
    assert formatted == result


def test_mpirun_args():
    run_args = {
        "map-by": "ppr:1:node",
        "np": 1,
    }
    settings = MpirunSettings("python", run_args=run_args)
    formatted = settings.format_run_args()
    result = ["--map-by", "ppr:1:node", "--np", "1"]
    assert formatted == result
    settings.set_task_map("ppr:2:node")
    formatted = settings.format_run_args()
    result = ["--map-by", "ppr:2:node", "--np", "1"]
    assert formatted == result


def test_mpirun_add_mpmd():
    settings = MpirunSettings("python")
    settings_2 = MpirunSettings("python")
    settings.make_mpmd(settings_2)
    assert len(settings.mpmd) > 0
    assert settings.mpmd[0] == settings_2

def test_catch_colo_mpmd():
    settings = MpirunSettings("python")
    settings.colocated_db_settings = {"port": 6379,
                                      "cpus": 1}
    settings_2 = MpirunSettings("python")
    with pytest.raises(SSUnsupportedError):
        settings.make_mpmd(settings_2)

def test_format_env():
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose"}
    settings = MpirunSettings("python", env_vars=env_vars)
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
    settings = MpirunSettings("python")
    with pytest.raises(TypeError):
        settings.set_hostlist(4)


def test_mpirun_hostlist_errors_1():
    settings = MpirunSettings("python")
    with pytest.raises(TypeError):
        settings.set_hostlist([444])
