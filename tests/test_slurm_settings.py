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
from smartsim.settings import SbatchSettings, SrunSettings

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


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
    num_threads = 10
    settings.update_env({"OMP_NUM_THREADS": num_threads})
    assert settings.env_vars["OMP_NUM_THREADS"] == str(num_threads)


def test_catch_colo_mpmd():
    srun = SrunSettings("python")
    srun.colocated_db_settings = {"port": 6379, "cpus": 1}
    srun_2 = SrunSettings("python")

    # should catch the user trying to make rs mpmd that already are colocated
    with pytest.raises(SSUnsupportedError):
        srun.make_mpmd(srun_2)


def test_mpmd_compound_env_exports():
    """
    Test that compound env vars are added to root env and exported
    to the correct sub-command in mpmd cmd
    """
    srun = SrunSettings("printenv")
    srun.in_batch = True
    srun.alloc = 12345
    srun.env_vars = {"cmp1": "123,456", "norm1": "xyz"}
    srun_2 = SrunSettings("printenv")
    srun_2.env_vars = {"cmp2": "222,333", "norm2": "pqr"}
    srun.make_mpmd(srun_2)

    from smartsim._core.launcher.step.slurmStep import SbatchStep, SrunStep
    from smartsim.settings.slurmSettings import SbatchSettings

    step = SrunStep("teststep", "./", srun)

    launch_cmd = step.get_launch_cmd()
    env_cmds = [v for v in launch_cmd if v == "env"]
    assert "env" in launch_cmd and len(env_cmds) == 1

    # ensure mpmd command is concatenated
    mpmd_delimiter_idx = launch_cmd.index(":")
    assert mpmd_delimiter_idx > -1

    # ensure root cmd exports env
    root_cmd = launch_cmd[:mpmd_delimiter_idx]
    exp_idx = root_cmd.index("--export")
    assert exp_idx

    # ensure correct exports
    env_vars = root_cmd[exp_idx + 1]
    assert "cmp1" in env_vars
    assert "norm1=xyz" in env_vars
    assert "cmp2" not in env_vars
    assert "norm2" not in env_vars

    # ensure mpmd cmd exports env
    mpmd_cmd = launch_cmd[mpmd_delimiter_idx:]
    exp_idx = mpmd_cmd.index("--export")
    assert exp_idx

    # ensure correct exports
    env_vars = mpmd_cmd[exp_idx + 1]
    assert "cmp2" in env_vars
    assert "norm2=pqr" in env_vars
    assert "cmp1" not in env_vars
    assert "norm1" not in env_vars

    srun_idx = " ".join(launch_cmd).index("srun")
    assert srun_idx > -1

    # ensure correct vars loaded in parent shell
    env_area = launch_cmd[:srun_idx]
    assert "env" in env_area
    assert "cmp1=123,456" in env_area
    assert "cmp2=222,333" in env_area


def test_mpmd_non_compound_env_exports():
    """
    Test that without compound env vars, no `env <k=v>...` is prepended to cmd
    """
    srun = SrunSettings("printenv")
    srun.in_batch = True
    srun.alloc = 12345
    srun.env_vars = {"cmp1": "123", "norm1": "xyz"}
    srun_2 = SrunSettings("printenv")
    srun_2.env_vars = {"cmp2": "222", "norm2": "pqr"}
    srun.make_mpmd(srun_2)

    from smartsim._core.launcher.step.slurmStep import SbatchStep, SrunStep
    from smartsim.settings.slurmSettings import SbatchSettings

    step = SrunStep("teststep", "./", srun)

    launch_cmd = step.get_launch_cmd()
    env_cmds = [v for v in launch_cmd if v == "env"]
    assert "env" not in launch_cmd and len(env_cmds) == 0

    # ensure mpmd command is concatenated
    mpmd_delimiter_idx = launch_cmd.index(":")
    assert mpmd_delimiter_idx > -1

    # ensure root cmd exports env
    root_cmd = launch_cmd[:mpmd_delimiter_idx]
    exp_idx = root_cmd.index("--export")
    assert exp_idx

    # ensure correct exports
    env_vars = root_cmd[exp_idx + 1]
    assert "cmp1" in env_vars
    assert "norm1=xyz" in env_vars
    assert "cmp2" not in env_vars
    assert "norm2" not in env_vars

    # ensure mpmd cmd exports env
    mpmd_cmd = launch_cmd[mpmd_delimiter_idx:]
    exp_idx = mpmd_cmd.index("--export")
    assert exp_idx

    # ensure correct exports
    env_vars = mpmd_cmd[exp_idx + 1]
    assert "cmp2" in env_vars
    assert "norm2=pqr" in env_vars
    assert "cmp1" not in env_vars
    assert "norm1" not in env_vars

    srun_idx = " ".join(launch_cmd).index("srun")
    assert srun_idx > -1

    # ensure correct vars loaded in parent shell
    env_area = launch_cmd[:srun_idx]
    assert "env" not in env_area
    assert "cmp1=123" not in env_area
    assert "cmp2=222" not in env_area


def test_mpmd_non_compound_no_exports():
    """
    Test that no --export is added if no env vars are supplied
    """
    srun = SrunSettings("printenv")
    srun.in_batch = True
    srun.alloc = 12345
    srun.env_vars = {}
    srun_2 = SrunSettings("printenv")
    srun_2.env_vars = {}
    srun.make_mpmd(srun_2)

    from smartsim._core.launcher.step.slurmStep import SbatchStep, SrunStep
    from smartsim.settings.slurmSettings import SbatchSettings

    step = SrunStep("teststep", "./", srun)

    launch_cmd = step.get_launch_cmd()
    env_cmds = [v for v in launch_cmd if v == "env"]
    assert "env" not in launch_cmd and len(env_cmds) == 0

    # ensure mpmd command is concatenated
    mpmd_delimiter_idx = launch_cmd.index(":")
    assert mpmd_delimiter_idx > -1

    # ensure no --export exists in either command
    assert "--export" not in launch_cmd


def test_format_env_vars():
    rs = SrunSettings(
        "python",
        env_vars={
            "OMP_NUM_THREADS": 20,
            "LOGGING": "verbose",
            "SSKEYIN": "name_0,name_1",
        },
    )
    formatted = rs.format_env_vars()
    assert "OMP_NUM_THREADS=20" in formatted
    assert "LOGGING=verbose" in formatted
    assert all("SSKEYIN" not in x for x in formatted)


def test_catch_existing_env_var(caplog, monkeypatch):
    rs = SrunSettings(
        "python",
        env_vars={
            "SMARTSIM_TEST_VAR": "B",
        },
    )
    monkeypatch.setenv("SMARTSIM_TEST_VAR", "A")
    monkeypatch.setenv("SMARTSIM_TEST_CSVAR", "A,B")
    caplog.clear()
    rs.format_env_vars()

    msg = f"Variable SMARTSIM_TEST_VAR is set to A in current environment. "
    msg += f"If the job is running in an interactive allocation, the value B will not be set. "
    msg += "Please consider removing the variable from the environment and re-running the experiment."

    for record in caplog.records:
        assert record.levelname == "WARNING"
        assert record.message == msg

    caplog.clear()

    env_vars = {"SMARTSIM_TEST_VAR": "B", "SMARTSIM_TEST_CSVAR": "C,D"}
    settings = SrunSettings("python", env_vars=env_vars)
    settings.format_comma_sep_env_vars()

    for record in caplog.records:
        assert record.levelname == "WARNING"
        assert record.message == msg


def test_format_comma_sep_env_vars():
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose", "SSKEYIN": "name_0,name_1"}
    settings = SrunSettings("python", env_vars=env_vars)
    formatted, comma_separated_formatted = settings.format_comma_sep_env_vars()
    assert "OMP_NUM_THREADS" in formatted
    assert "LOGGING" in formatted
    assert "SSKEYIN" in formatted
    assert "name_0,name_1" not in formatted
    assert "SSKEYIN=name_0,name_1" in comma_separated_formatted


@pytest.mark.parametrize("reserved_arg", ["chdir", "D"])
def test_no_set_reserved_args(reserved_arg):
    srun = SrunSettings("python")
    srun.set(reserved_arg)
    assert reserved_arg not in srun.run_args


def test_set_tasks():
    rs = SrunSettings("python")
    rs.set_tasks(6)
    assert rs.run_args["ntasks"] == 6

    with pytest.raises(ValueError):
        rs.set_tasks("not an int")


def test_set_tasks_per_node():
    rs = SrunSettings("python")
    rs.set_tasks_per_node(6)
    assert rs.run_args["ntasks-per-node"] == 6

    with pytest.raises(ValueError):
        rs.set_tasks_per_node("not an int")


def test_set_cpus_per_task():
    rs = SrunSettings("python")
    rs.set_cpus_per_task(6)
    assert rs.run_args["cpus-per-task"] == 6

    with pytest.raises(ValueError):
        rs.set_cpus_per_task("not an int")


def test_set_hostlist():
    rs = SrunSettings("python")
    rs.set_hostlist(["host_A", "host_B"])
    assert rs.run_args["nodelist"] == "host_A,host_B"

    rs.set_hostlist("host_A")
    assert rs.run_args["nodelist"] == "host_A"

    with pytest.raises(TypeError):
        rs.set_hostlist([5])


def test_set_hostlist_from_file():
    rs = SrunSettings("python")
    rs.set_hostlist_from_file("./path/to/hostfile")
    assert rs.run_args["nodefile"] == "./path/to/hostfile"

    rs.set_hostlist_from_file("~/other/file")
    assert rs.run_args["nodefile"] == "~/other/file"


def test_set_cpu_bindings():
    rs = SrunSettings("python")
    rs.set_cpu_bindings([1, 2, 3, 4])
    assert rs.run_args["cpu_bind"] == "map_cpu:1,2,3,4"

    rs.set_cpu_bindings(2)
    assert rs.run_args["cpu_bind"] == "map_cpu:2"

    with pytest.raises(ValueError):
        rs.set_cpu_bindings(["not_an_int"])


def test_set_memory_per_node():
    rs = SrunSettings("python")
    rs.set_memory_per_node(8000)
    assert rs.run_args["mem"] == "8000M"

    with pytest.raises(ValueError):
        rs.set_memory_per_node("not_an_int")


def test_set_verbose():
    rs = SrunSettings("python")
    rs.set_verbose_launch(True)
    assert "verbose" in rs.run_args

    rs.set_verbose_launch(False)
    assert "verbose" not in rs.run_args

    # Ensure not error on repeat calls
    rs.set_verbose_launch(False)


def test_quiet_launch():
    rs = SrunSettings("python")
    rs.set_quiet_launch(True)
    assert "quiet" in rs.run_args

    rs.set_quiet_launch(False)
    assert "quiet" not in rs.run_args

    # Ensure not error on repeat calls
    rs.set_quiet_launch(False)


def test_set_broadcast():
    rs = SrunSettings("python")
    rs.set_broadcast("/tmp/some/path")
    assert rs.run_args["bcast"] == "/tmp/some/path"


def test_set_time():
    rs = SrunSettings("python")
    rs.set_time(seconds=72)
    assert rs.run_args["time"] == "00:01:12"

    rs.set_time(hours=1, minutes=31, seconds=1845)
    assert rs.run_args["time"] == "02:01:45"

    rs.set_time(hours=11)
    assert rs.run_args["time"] == "11:00:00"

    rs.set_time(seconds=0)
    assert rs.run_args["time"] == "00:00:00"

    with pytest.raises(ValueError):
        rs.set_time("not an int")


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
