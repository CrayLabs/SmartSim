import pytest

from smartsim.settings import BatchSettings
from smartsim.settings.batchCommand import SchedulerType


@pytest.mark.parametrize(
    "scheduler_enum",
    [
        pytest.param(SchedulerType.Slurm, id="slurm"),
        pytest.param(SchedulerType.Pbs, id="dragon"),
        pytest.param(SchedulerType.Lsf, id="lsf"),
    ],
)
def test_create_scheduler_settings(scheduler_enum):
    bs_str = BatchSettings(
        batch_scheduler=scheduler_enum.value,
        scheduler_args={"launch": "var"},
        env_vars={"ENV": "VAR"},
    )
    print(bs_str)
    assert bs_str._batch_scheduler == scheduler_enum
    # TODO need to test scheduler_args
    assert bs_str._env_vars == {"ENV": "VAR"}

    bs_enum = BatchSettings(
        batch_scheduler=scheduler_enum,
        scheduler_args={"launch": "var"},
        env_vars={"ENV": "VAR"},
    )
    assert bs_enum._batch_scheduler == scheduler_enum
    # TODO need to test scheduler_args
    assert bs_enum._env_vars == {"ENV": "VAR"}


def test_launcher_property():
    bs = BatchSettings(batch_scheduler="slurm")
    assert bs.batch_scheduler == "slurm"


def test_env_vars_property():
    bs = BatchSettings(batch_scheduler="slurm", env_vars={"ENV": "VAR"})
    assert bs.env_vars == {"ENV": "VAR"}


def test_env_vars_property_deep_copy():
    bs = BatchSettings(batch_scheduler="slurm", env_vars={"ENV": "VAR"})
    copy_env_vars = bs.env_vars
    copy_env_vars.update({"test": "no_update"})
    assert bs.env_vars == {"ENV": "VAR"}
