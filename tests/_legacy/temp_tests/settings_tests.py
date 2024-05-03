from smartsim.settings import RunSettings, SrunSettings, PalsMpiexecSettings, MpirunSettings, MpiexecSettings, OrterunSettings, JsrunSettings, AprunSettings, BsubBatchSettings, QsubBatchSettings, SbatchSettings
import os
from shutil import which
import pytest
import itertools
import os.path as osp

env_vars = {"k1": "v1", "k2": "v2"}
run_args = {"envlist": "SPAM"}

# Test that mpi RunSetting classes create without error
@pytest.mark.parametrize(
    "settings_type, env_vars, run_args",
    [
        pytest.param(
            MpirunSettings,
            env_vars,
            run_args,
            id=f"mpirun",
        ),
        pytest.param(
            OrterunSettings,
            env_vars,
            run_args,
            id=f"orterun",
        )
    ]
)
def test_mpi_instantiate_run_settings(
    settings_type, env_vars, run_args
):
    settings = settings_type(run_args=run_args, env_vars=env_vars, fail_if_missing_exec=False)
    assert settings.env_vars == env_vars
    assert settings.run_args == run_args
    assert isinstance(settings, settings_type)

# Test that RunSetting classes create without error
@pytest.mark.parametrize(
    "settings_type, env_vars, run_args",
    [
        pytest.param(
            SrunSettings,
            env_vars,
            run_args,
            id=f"srun",
        ),
        pytest.param(
            PalsMpiexecSettings,
            env_vars,
            run_args,
            id=f"mpiexec",
        ),
        pytest.param(
            JsrunSettings,
            env_vars,
            run_args,
            id="jsrun",
        ),
        pytest.param(
            RunSettings,
            env_vars,
            run_args,
            id="local",
        ),
        pytest.param(
            AprunSettings,
            env_vars,
            run_args,
            id="aprun",
        )
    ]
)
def test_instantiate_run_settings(
    settings_type, env_vars, run_args
):
    settings = settings_type(run_args=run_args, env_vars=env_vars)
    assert settings.env_vars == env_vars
    assert settings.run_args == run_args
    assert isinstance(settings, settings_type)

nodes = 4
time = "10:00:00"
account = "1234"

# Test that BatchSettings classes create without error
# This currently does not work, need to unify how we treat each settings class
@pytest.mark.parametrize(
    "settings_type, nodes, node_flag, time, account",
    [
        pytest.param(
            BsubBatchSettings,
            nodes,
            "nnodes",
            time,
            account,
            id=f"bsub",
        ),
        pytest.param(
            QsubBatchSettings,
            nodes,
            "nodes",
            time,
            account,
            id="qsub",
        ),
        pytest.param(
            SbatchSettings,
            nodes,
            "nodes",
            time,
            account,
            id="sbatch",
        )
    ]
)
def test_instantiate_batch_settings(
    settings_type, nodes, node_flag, time, account
):
    batch_settings = settings_type(nodes=nodes, time=time, account=account)
    assert batch_settings.resources[node_flag] == nodes
    assert batch_settings.batch_args["time"] == time
    assert batch_settings.batch_args["account"] == account
    assert isinstance(batch_settings, settings_type)