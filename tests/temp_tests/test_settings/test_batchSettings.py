from smartsim.settings import BatchSettings
from smartsim.settings.batchCommand import SchedulerType
import pytest

@pytest.mark.parametrize(
    "str,scheduler",
    [
        pytest.param("slurm", SchedulerType.SlurmScheduler, id="slurm"),
        pytest.param("pbs", SchedulerType.PbsScheduler, id="dragon"),
        pytest.param("lsf", SchedulerType.LsfScheduler, id="lsf"),
    ],
)
def test_create_from_scheduler_str(str, scheduler):
    batchSettings = BatchSettings(batch_scheduler=str)
    assert batchSettings.batch_scheduler == scheduler

def test_incorrect_env_var_type():
    with pytest.raises(TypeError):
        _ = BatchSettings(batch_scheduler=SchedulerType.SlurmScheduler, env_vars={"str": 2})
    with pytest.raises(TypeError):
        _ = BatchSettings(batch_scheduler=SchedulerType.SlurmScheduler, env_vars={"str": 2.0})
    with pytest.raises(TypeError):
        _ = BatchSettings(batch_scheduler=SchedulerType.SlurmScheduler, env_vars={"str": "str", "str": 2.0})

def test_incorrect_scheduler_arg_type():
    with pytest.raises(TypeError):
        _ = BatchSettings(batch_scheduler=SchedulerType.SlurmScheduler, scheduler_args={"str": [1,2]})
    with pytest.raises(TypeError):
        _ = BatchSettings(batch_scheduler=SchedulerType.SlurmScheduler, scheduler_args={"str": SchedulerType.SlurmScheduler})