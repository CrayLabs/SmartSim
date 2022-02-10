import pytest

from smartsim.settings import SbatchSettings, SrunSettings
from smartsim.error import SSUnsupportedError

# ------ Srun ------------------------------------------------


def test_srun_settings():
    settings = SrunSettings("python")
    settings.set_nodes(5)
    settings.set_cpus_per_task(2)
    settings.set_tasks(100)
    settings.set_tasks_per_node(20)
    formatted = settings.format_run_args()
    result = ["--nodes=5", "--cpus-per-task=2", "--ntasks=100", "--ntasks-per-node=20"]
    assert formatted == result


def test_srun_args():
    """Test the possible user overrides through run_args"""
    run_args = {
        "account": "A3123",
        "exclusive": None,
        "C": "P100",  # test single letter variables
        "nodes": 10,
        "ntasks": 100,
    }
    settings = SrunSettings("python", run_args=run_args)
    formatted = settings.format_run_args()
    result = [
        "--account=A3123",
        "--exclusive",
        "-C",
        "P100",
        "--nodes=10",
        "--ntasks=100",
    ]
    assert formatted == result


def test_update_env():
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose"}
    settings = SrunSettings("python", env_vars=env_vars)
    settings.update_env({"OMP_NUM_THREADS": 10})
    assert settings.env_vars["OMP_NUM_THREADS"] == 10


def test_catch_colo_mpmd():
    srun = SrunSettings("python")
    srun.colocated_db_settings = {"port": 6379,
                                  "cpus": 1}
    srun_2 = SrunSettings("python")

    # should catch the user trying to make rs mpmd that already are colocated
    with pytest.raises(SSUnsupportedError):
        srun.make_mpmd(srun_2)


def test_format_env():
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose", "SSKEYIN": "name_0,name_1"}
    settings = SrunSettings("python", env_vars=env_vars)
    formatted, comma_separated_formatted = settings.format_env_vars()
    assert "OMP_NUM_THREADS" in formatted
    assert "LOGGING" in formatted
    assert "SSKEYIN" in formatted
    assert "SSKEYIN=name_0,name_1" in comma_separated_formatted


# ---- Sbatch ---------------------------------------------------


def test_sbatch_settings():
    sbatch = SbatchSettings(nodes=1, time="10:00:00", account="A3123")
    formatted = sbatch.format_batch_args()
    result = ["--nodes=1", "--time=10:00:00", "--account=A3123"]
    assert formatted == result


def test_sbatch_manual():
    sbatch = SbatchSettings()
    sbatch.set_nodes(5)
    sbatch.set_account("A3531")
    sbatch.set_walltime("10:00:00")
    formatted = sbatch.format_batch_args()
    result = ["--nodes=5", "--account=A3531", "--time=10:00:00"]
    assert formatted == result


def test_change_batch_cmd():
    sbatch = SbatchSettings()
    sbatch.set_batch_command("qsub")
    assert sbatch._batch_cmd == "qsub"
