from pprint import pformat

import pytest

from smartsim.settings import BsubBatchSettings, JsrunSettings
from smartsim.error import SSUnsupportedError

# ------ Jsrun ------------------------------------------------


def test_jsrun_settings():
    settings = JsrunSettings("python")
    settings.set_num_rs(8)
    settings.set_cpus_per_rs(2)
    settings.set_gpus_per_rs(1)
    settings.set_rs_per_host(4)
    settings.set_tasks_per_rs(12)
    settings.set_tasks(96)
    settings.set_binding("packed:2")
    formatted = settings.format_run_args()
    result = [
        "--nrs=8",
        "--cpu_per_rs=2",
        "--gpu_per_rs=1",
        "--rs_per_host=4",
        "--tasks_per_rs=12",
        "--np=96",
        "--bind=packed:2",
    ]
    assert formatted == result

    settings.set_cpus_per_rs("ALL_CPUS")
    settings.set_gpus_per_rs("ALL_GPUS")
    settings.set_num_rs("ALL_HOSTS")

    formatted = settings.format_run_args()
    result = [
        "--nrs=ALL_HOSTS",
        "--cpu_per_rs=ALL_CPUS",
        "--gpu_per_rs=ALL_GPUS",
        "--rs_per_host=4",
        "--tasks_per_rs=12",
        "--np=96",
        "--bind=packed:2",
    ]
    assert formatted == result


def test_jsrun_args():
    """Test the possible user overrides through run_args"""
    run_args = {
        "latency_priority": "gpu-gpu",
        "immediate": None,
        "d": "packed",  # test single letter variables
        "nrs": 10,
        "np": 100,
    }
    settings = JsrunSettings("python", run_args=run_args)
    formatted = settings.format_run_args()
    result = [
        "--latency_priority=gpu-gpu",
        "--immediate",
        "-d",
        "packed",
        "--nrs=10",
        "--np=100",
    ]
    assert formatted == result


def test_jsrun_update_env():
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose"}
    settings = JsrunSettings("python", env_vars=env_vars)
    settings.update_env({"OMP_NUM_THREADS": 10})
    assert settings.env_vars["OMP_NUM_THREADS"] == 10


def test_jsrun_format_env():
    # Test propagation (no value setting)
    env_vars = {"OMP_NUM_THREADS": None, "LOGGING": "verbose"}
    settings = JsrunSettings("python", env_vars=env_vars)
    formatted = settings.format_env_vars()
    assert formatted == ["-E", "OMP_NUM_THREADS", "-E", "LOGGING=verbose"]


def test_jsrun_mpmd():
    settings = JsrunSettings("python")
    settings.set_mpmd_preamble(["launch_distribution : packed"])
    assert settings.mpmd_preamble_lines == ["launch_distribution : packed"]


def test_catch_colo_mpmd():
    settings = JsrunSettings("python")
    settings.colocated_db_settings = {"port": 6379,
                                      "cpus": 1}
    settings_2 = JsrunSettings("python")
    with pytest.raises(SSUnsupportedError):
        settings.make_mpmd(settings_2)

# ---- Bsub Batch ---------------------------------------------------


def test_bsub_batch_settings():
    sbatch = BsubBatchSettings(
        nodes=1,
        time="10:00:00",
        project="A3123",
        smts=4,
        batch_args={"alloc_flags": "nvme"},
    )
    formatted = sbatch.format_batch_args()
    result = ['-alloc_flags "nvme smt4"', "-nnodes 1"]
    assert formatted == result


def test_bsub_batch_manual():
    sbatch = BsubBatchSettings(batch_args={"alloc_flags": "gpumps smt4"})
    sbatch.set_nodes(5)
    sbatch.set_project("A3531")
    sbatch.set_walltime("10:00:00")
    sbatch._format_alloc_flags()
    # Enclose in quotes if user did not
    assert sbatch.batch_args["alloc_flags"] == '"gpumps smt4"'
    sbatch.set_smts("2")  # This should have no effect as per our docs
    sbatch.set_hostlist(["node1", "node2", "node5"])
    sbatch.set_tasks(5)
    formatted = sbatch.format_batch_args()
    result = [
        '-alloc_flags "gpumps smt4"',
        "-nnodes 5",
        '-m "node1 node2 node5"',
        "-n 5",
    ]
    assert formatted == result
    sbatch.add_preamble("module load gcc")
    sbatch.add_preamble(["module load openmpi", "conda activate smartsim"])
    assert sbatch._preamble == [
        "module load gcc",
        "module load openmpi",
        "conda activate smartsim",
    ]

    with pytest.raises(TypeError):
        sbatch.add_preamble(1)
