import pytest

from smartsim.settings import BatchSettings
from smartsim.settings.batchCommand import SchedulerType


def test_scheduler_str():
    """Ensure scheduler_str returns appropriate value"""
    bs = BatchSettings(batch_scheduler=SchedulerType.Lsf)
    assert bs.scheduler_args.scheduler_str() == SchedulerType.Lsf.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,), "2", "nnodes", id="set_nodes"),
        pytest.param("set_walltime", ("10:00:00",), "10:00", "W", id="set_walltime"),
        pytest.param(
            "set_hostlist", ("host_A",), "" '"host_A"' "", "m", id="set_hostlist_str"
        ),
        pytest.param(
            "set_hostlist",
            (["host_A", "host_B"],),
            "" '"host_A host_B"' "",
            "m",
            id="set_hostlist_list[str]",
        ),
        pytest.param("set_smts", (1,), "1", "alloc_flags", id="set_smts"),
        pytest.param("set_project", ("project",), "project", "P", id="set_project"),
        pytest.param("set_account", ("project",), "project", "P", id="set_account"),
        pytest.param("set_tasks", (2,), "2", "n", id="set_tasks"),
        pytest.param("set_queue", ("queue",), "queue", "q", id="set_queue"),
    ],
)
def test_update_env_initialized(function, value, flag, result):
    lsfScheduler = BatchSettings(batch_scheduler=SchedulerType.Lsf)
    getattr(lsfScheduler.scheduler_args, function)(*value)
    assert lsfScheduler.scheduler_args._scheduler_args[flag] == result


def test_create_bsub():
    batch_args = {"core_isolation": None}
    lsfScheduler = BatchSettings(
        batch_scheduler=SchedulerType.Lsf, scheduler_args=batch_args
    )
    lsfScheduler.scheduler_args.set_nodes(1)
    lsfScheduler.scheduler_args.set_walltime("10:10:10")
    lsfScheduler.scheduler_args.set_queue("default")
    args = lsfScheduler.format_batch_args()
    assert args == ["-core_isolation", "-nnodes", "1", "-W", "10:10", "-q", "default"]
