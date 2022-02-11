import pytest
from smartsim.settings import AprunSettings
from smartsim.error import SSUnsupportedError

def test_aprun_settings():
    settings = AprunSettings("python")
    settings.set_cpus_per_task(2)
    settings.set_tasks(100)
    settings.set_tasks_per_node(20)
    formatted = settings.format_run_args()
    result = ["--cpus-per-pe=2", "--pes=100", "--pes-per-node=20"]
    assert formatted == result


def test_aprun_args():
    run_args = {
        "z": None,  # single letter no value
        "sync-output": None,  # long no value
        "wdir": "/some/path",  # restricted var
        "pes": 5,
    }
    settings = AprunSettings("python", run_args=run_args)
    formatted = settings.format_run_args()
    result = ["-z", "--sync-output", "--pes=5"]
    assert formatted == result


def test_aprun_add_mpmd():
    settings = AprunSettings("python")
    settings_2 = AprunSettings("python")
    settings.make_mpmd(settings_2)
    assert len(settings.mpmd) > 0
    assert settings.mpmd[0] == settings_2

def test_catch_colo_mpmd():
    settings = AprunSettings("python")
    settings.colocated_db_settings = {"port": 6379,
                                      "cpus": 1}
    settings_2 = AprunSettings("python")
    with pytest.raises(SSUnsupportedError):
        settings.make_mpmd(settings_2)


def test_format_env():
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose"}
    settings = AprunSettings("python", env_vars=env_vars)
    settings.update_env({"OMP_NUM_THREADS": 10})
    formatted = settings.format_env_vars()
    result = ["-e", "OMP_NUM_THREADS=10", "-e", "LOGGING=verbose"]
    assert formatted == result
