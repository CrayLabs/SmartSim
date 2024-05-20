from smartsim.settingshold import BatchSettings
from smartsim.settingshold.batchCommand import SchedulerType
import pytest

def test_incorrect_env_var_type():
    with pytest.raises(TypeError):
        _ = BatchSettings(launcher=SchedulerType.SlurmScheduler, env_vars={"str": 2})
    with pytest.raises(TypeError):
        _ = BatchSettings(launcher=SchedulerType.SlurmScheduler, env_vars={"str": 2.0})
    with pytest.raises(TypeError):
        _ = BatchSettings(launcher=SchedulerType.SlurmScheduler, env_vars={"str": "str", "str": 2.0})

def test_incorrect_scheduler_arg_type():
    with pytest.raises(TypeError):
        _ = BatchSettings(launcher=SchedulerType.SlurmScheduler, launcher_args={"str": [1,2]})
    with pytest.raises(TypeError):
        _ = BatchSettings(launcher=SchedulerType.SlurmScheduler, launcher_args={"str": SchedulerType.SlurmScheduler})