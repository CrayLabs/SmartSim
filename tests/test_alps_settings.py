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
    settings.colocated_db_settings = {"port": 6379, "cpus": 1}
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


def test_set_cpus_per_task():
    rs = AprunSettings("python")
    rs.set_cpus_per_task(6)
    assert rs.run_args["cpus-per-pe"] == 6

    with pytest.raises(ValueError):
        rs.set_cpus_per_task("not an int")


def test_set_tasks():
    rs = AprunSettings("python")
    rs.set_tasks(6)
    assert rs.run_args["pes"] == 6

    with pytest.raises(ValueError):
        rs.set_tasks("not an int")


def test_set_tasks_per_node():
    rs = AprunSettings("python")
    rs.set_tasks_per_node(6)
    assert rs.run_args["pes-per-node"] == 6

    with pytest.raises(ValueError):
        rs.set_tasks_per_node("not an int")


def test_set_hostlist():
    rs = AprunSettings("python")
    rs.set_hostlist(["host_A", "host_B"])
    assert rs.run_args["node-list"] == "host_A,host_B"

    rs.set_hostlist("host_A")
    assert rs.run_args["node-list"] == "host_A"

    with pytest.raises(TypeError):
        rs.set_hostlist([5])


def test_set_cpu_bindings():
    rs = AprunSettings("python")
    rs.set_cpu_bindings([1, 2, 3, 4])
    assert rs.run_args["cpu-binding"] == "1,2,3,4"

    rs.set_cpu_bindings(2)
    assert rs.run_args["cpu-binding"] == "2"

    with pytest.raises(ValueError):
        rs.set_cpu_bindings(["not_an_int"])


def test_set_memory_per_node():
    rs = AprunSettings("python")
    rs.set_memory_per_node(8000)
    assert rs.run_args["memory-per-pe"] == 8000

    with pytest.raises(ValueError):
        rs.set_memory_per_node("not_an_int")


def test_set_verbose():
    rs = AprunSettings("python")
    rs.set_verbose_launch(True)
    assert "debug" in rs.run_args
    assert rs.run_args["debug"] == 7

    rs.set_verbose_launch(False)
    assert "debug" not in rs.run_args

    # Ensure not error on repeat calls
    rs.set_verbose_launch(False)


def test_quiet_launch():
    rs = AprunSettings("python")
    rs.set_quiet_launch(True)
    assert "quiet" in rs.run_args

    rs.set_quiet_launch(False)
    assert "quiet" not in rs.run_args

    # Ensure not error on repeat calls
    rs.set_quiet_launch(False)


def test_set_timeout():
    rs = AprunSettings("python")
    rs.set_timeout(72)
    assert rs.run_args["cpu-time-limit"] == 72

    rs.set_timeout(0)
    assert rs.run_args["cpu-time-limit"] == 0

    with pytest.raises(ValueError):
        rs.set_timeout("not an int")
