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


import pytest

from smartsim.error import SSUnsupportedError
from smartsim.settings import BsubBatchSettings, JsrunSettings

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b

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


def test_jsrun_args_mutation():
    """Ensure re-using ERF settings doesn't mutate existing run settings"""
    run_args = {
        "latency_priority": "gpu-gpu",
        "immediate": None,
        "d": "packed",  # test single letter variables
        "nrs": 10,
        "np": 100,
    }
    settings = JsrunSettings("python", run_args=run_args)

    erf_settings = {"foo": "1", "bar": "2"}

    settings.set_erf_sets(erf_settings)
    assert settings.erf_sets["foo"] == "1"
    assert settings.erf_sets["bar"] == "2"

    erf_settings["foo"] = "111"
    erf_settings["bar"] = "111"

    assert settings.erf_sets["foo"] == "1"
    assert settings.erf_sets["bar"] == "2"


def test_jsrun_update_env():
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose"}
    settings = JsrunSettings("python", env_vars=env_vars)
    num_threads = 10
    settings.update_env({"OMP_NUM_THREADS": num_threads})
    assert settings.env_vars["OMP_NUM_THREADS"] == str(num_threads)


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
    settings.colocated_db_settings = {"port": 6379, "cpus": 1}
    settings_2 = JsrunSettings("python")
    with pytest.raises(SSUnsupportedError):
        settings.make_mpmd(settings_2)


@pytest.mark.parametrize("reserved_arg", ["chdir", "h"])
def test_no_set_reserved_args(reserved_arg):
    srun = JsrunSettings("python")
    srun.set(reserved_arg)
    assert reserved_arg not in srun.run_args


def test_set_tasks():
    rs = JsrunSettings("python")
    rs.set_tasks(6)
    assert rs.run_args["np"] == 6

    with pytest.raises(ValueError):
        rs.set_tasks("not an int")


def test_set_tasks_per_node():
    rs = JsrunSettings("python")
    rs.set_tasks_per_node(6)
    assert rs.run_args["tasks_per_rs"] == 6

    with pytest.raises(ValueError):
        rs.set_tasks_per_node("not an int")


def test_set_cpus_per_task():
    rs = JsrunSettings("python")
    rs.set_cpus_per_task(6)
    assert rs.run_args["cpu_per_rs"] == 6

    with pytest.raises(ValueError):
        rs.set_cpus_per_task("not an int")


def test_set_memory_per_node():
    rs = JsrunSettings("python")
    rs.set_memory_per_node(8000)
    assert rs.run_args["memory_per_rs"] == 8000

    with pytest.raises(ValueError):
        rs.set_memory_per_node("not_an_int")


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
    assert list(sbatch.preamble) == [
        "module load gcc",
        "module load openmpi",
        "conda activate smartsim",
    ]

    with pytest.raises(TypeError):
        sbatch.add_preamble(1)


def test_bsub_batch_alloc_flag_formatting_by_smt():
    """Ensure that alloc_flags are formatted correctly when smts is changed"""

    # Check when no smt is set in the constructor
    sbatch = BsubBatchSettings()
    sbatch._format_alloc_flags()
    assert "alloc_flags" not in sbatch.batch_args

    # check when using set_smts
    sbatch = BsubBatchSettings(smts=2)
    sbatch._format_alloc_flags()
    assert "alloc_flags" in sbatch.batch_args
    assert sbatch.batch_args["alloc_flags"] == "smt2"

    # Check when passing alloc_flags in constructor
    sbatch = BsubBatchSettings(batch_args={"alloc_flags": "unittest-smt"}, smts=0)
    sbatch._format_alloc_flags()
    assert sbatch.batch_args["alloc_flags"] == "unittest-smt"

    # if smts=(non-zero), smt is *not* prepended to alloc_flags
    sbatch = BsubBatchSettings(batch_args={"alloc_flags": "unittest-smt"})
    sbatch._format_alloc_flags()
    assert sbatch.batch_args["alloc_flags"] == "unittest-smt"

    # Check when passing only SMT in constructor
    sbatch = BsubBatchSettings(smts=1)
    sbatch._format_alloc_flags()
    assert sbatch.batch_args["alloc_flags"] == "smt1"

    # Check prepending smt to alloc_flags value
    sbatch = BsubBatchSettings(atch_args={"alloc_flags": "3"}, smts=3)
    sbatch._format_alloc_flags()
    assert sbatch.batch_args["alloc_flags"] == "smt3"

    # check multi-smt flag, with prefix
    sbatch = BsubBatchSettings(batch_args={"alloc_flags": '"smt3 smt4"'}, smts=4)
    sbatch._format_alloc_flags()
    assert sbatch.batch_args["alloc_flags"] == '"smt3 smt4"'  # <-- wrap in quotes

    # show that mismatched alloc_flags and smts are NOT touched
    sbatch = BsubBatchSettings(batch_args={"alloc_flags": "smt10"}, smts=2)
    sbatch._format_alloc_flags()
    assert sbatch.batch_args["alloc_flags"] == "smt10"  # <-- not smt2
