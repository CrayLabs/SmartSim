from smartsim.settingshold import BatchSettings
import pytest
import logging
from smartsim.settingshold.batchCommand import SchedulerType

def test_scheduler_str():
    """Ensure launcher_str returns appropriate value"""
    lsfScheduler = BatchSettings(scheduler=SchedulerType.LsfScheduler)
    assert lsfScheduler.scheduler_str() == SchedulerType.LsfScheduler.value

@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,),2,"nnodes",id="set_nodes"),
        pytest.param("set_walltime", ("10:00:00",),"10:00","W",id="set_walltime"),
        pytest.param("set_hostlist", ("host_A",),""'"host_A"'"","m",id="set_hostlist_str"),
        pytest.param("set_hostlist", (["host_A","host_B"],),""'"host_A host_B"'"","m",id="set_hostlist_list[str]"),
        pytest.param("set_smts", (1,),1,"alloc_flags",id="set_smts"),
        pytest.param("set_project", ("project",),"project","P",id="set_project"),
        pytest.param("set_account", ("project",),"project","P",id="set_account"),
        pytest.param("set_tasks", (2,),2,"n",id="set_tasks"),
        pytest.param("set_queue", ("queue",),"queue","q",id="set_queue"),
    ],
)
def test_update_env_initialized(function, value, flag, result):
    lsfScheduler = BatchSettings(scheduler=SchedulerType.LsfScheduler)
    getattr(lsfScheduler, function)(*value)
    assert lsfScheduler.scheduler_args[flag] == result

def test_create_bsub():
    batch_args = {"core_isolation": None}
    lsfScheduler = BatchSettings(scheduler=SchedulerType.LsfScheduler, scheduler_args=batch_args)
    lsfScheduler.set_nodes(1)
    lsfScheduler.set_walltime("10:10:10")
    lsfScheduler.set_queue("default")
    args = lsfScheduler.format_batch_args()
    assert args == ["-core_isolation", "-nnodes 1", "-W 10:10", "-q default"]

@pytest.mark.parametrize(
    "method,params",
    [
        pytest.param("set_partition", (3,), id="set_tasks"),
        pytest.param("set_cpus_per_task", (10,), id="set_smts"),
        pytest.param("set_ncpus", (2,), id="set_ncpus"),
    ],
)
def test_unimplimented_setters_throw_warning(caplog, method, params):
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        slurmScheduler = BatchSettings(scheduler=SchedulerType.LsfScheduler)
        try:
            getattr(slurmScheduler, method)(*params)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and (method and "not supported" and "bsub") in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )