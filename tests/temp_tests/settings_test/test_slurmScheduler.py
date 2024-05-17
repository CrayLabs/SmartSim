from smartsim.settingshold import BatchSettings
from smartsim.settingshold.translators.batch.slurm import SlurmBatchArgTranslator
from smartsim.settingshold.batchCommand import SchedulerType
import pytest
import logging

def test_scheduler_str():
    """Ensure launcher_str returns appropriate value"""
    slurmScheduler = BatchSettings(scheduler=SchedulerType.SlurmScheduler)
    assert slurmScheduler.scheduler_str() == SchedulerType.SlurmScheduler.value

@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,),2,"nodes",id="set_nodes"),
        pytest.param("set_walltime", ("10:00:00",),"10:00:00","time",id="set_walltime"),
        pytest.param("set_account", ("account",),"account","account",id="set_account"),
        pytest.param("set_partition", ("partition",),"partition","partition",id="set_partition"),
        pytest.param("set_queue", ("partition",),"partition","partition",id="set_queue"),
        pytest.param("set_cpus_per_task", (2,),2,"cpus-per-task",id="set_cpus_per_task"),
        pytest.param("set_hostlist", ("host_A",),"host_A","nodelist",id="set_hostlist_str"),
        pytest.param("set_hostlist", (["host_A","host_B"],),"host_A,host_B","nodelist",id="set_hostlist_list[str]"),
    ],
)
def test_update_env_initialized(function, value, flag, result):
    slurmScheduler = BatchSettings(scheduler=SchedulerType.SlurmScheduler)
    getattr(slurmScheduler, function)(*value)
    assert slurmScheduler.scheduler_args[flag] == result

def test_create_sbatch():
    batch_args = {"exclusive": None, "oversubscribe": None}
    slurmScheduler = BatchSettings(scheduler=SchedulerType.SlurmScheduler, scheduler_args=batch_args)
    assert isinstance(slurmScheduler.arg_translator, SlurmBatchArgTranslator)
    #assert slurmScheduler.batch_args["partition"] == "default"
    args = slurmScheduler.format_batch_args()
    assert args == [
        "--exclusive",
        "--oversubscribe"
    ]

def test_launch_args_input_mutation():
    # Tests that the run args passed in are not modified after initialization
    key0, key1, key2 = "arg0", "arg1", "arg2"
    val0, val1, val2 = "val0", "val1", "val2"

    default_scheduler_args = {
        key0: val0,
        key1: val1,
        key2: val2,
    }
    slurmScheduler = BatchSettings(scheduler=SchedulerType.SlurmScheduler, scheduler_args=default_scheduler_args)

    # Confirm initial values are set
    assert slurmScheduler.scheduler_args[key0] == val0
    assert slurmScheduler.scheduler_args[key1] == val1
    assert slurmScheduler.scheduler_args[key2] == val2

    # Update our common run arguments
    val2_upd = f"not-{val2}"
    default_scheduler_args[key2] = val2_upd

    # Confirm previously created run settings are not changed
    assert slurmScheduler.scheduler_args[key2] == val2

def test_sbatch_settings():
    scheduler_args = {"nodes": 1, "time": "10:00:00", "account": "A3123"}
    slurmScheduler = BatchSettings(scheduler=SchedulerType.SlurmScheduler,scheduler_args=scheduler_args)
    formatted = slurmScheduler.format_batch_args()
    result = ["--nodes=1", "--time=10:00:00", "--account=A3123"]
    assert formatted == result


def test_sbatch_manual():
    slurmScheduler = BatchSettings(scheduler=SchedulerType.SlurmScheduler)
    slurmScheduler.set_nodes(5)
    slurmScheduler.set_account("A3531")
    slurmScheduler.set_walltime("10:00:00")
    formatted = slurmScheduler.format_batch_args()
    result = ["--nodes=5", "--account=A3531", "--time=10:00:00"]
    assert formatted == result

@pytest.mark.parametrize(
    "method,params",
    [
        pytest.param("set_tasks", (3,), id="set_tasks"),
        pytest.param("set_smts", ("smts",), id="set_smts"),
        pytest.param("set_ncpus", (2,), id="set_ncpus"),
        pytest.param("set_project", ("project",), id="set_project"),
    ],
)
def test_unimplimented_setters_throw_warning(caplog, method, params):
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        slurmScheduler = BatchSettings(scheduler=SchedulerType.SlurmScheduler)
        try:
            getattr(slurmScheduler, method)(*params)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and (method and "not supported" and "sbatch") in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )