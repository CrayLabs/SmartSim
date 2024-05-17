from smartsim.settingshold import BatchSettings
from smartsim.settingshold.translators.batch.pbs import QsubBatchArgTranslator
from smartsim.settingshold.batchCommand import SchedulerType
import pytest
import logging

def test_scheduler_str():
    """Ensure launcher_str returns appropriate value"""
    pbsScheduler = BatchSettings(scheduler=SchedulerType.PbsScheduler)
    assert pbsScheduler.scheduler_str() == SchedulerType.PbsScheduler.value

@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,),2,"nodes",id="set_nodes"),
        pytest.param("set_walltime", ("10:00:00",),"10:00:00","walltime",id="set_walltime"),
        pytest.param("set_account", ("account",),"account","A",id="set_account"),
        pytest.param("set_queue", ("queue",),"queue","q",id="set_queue"),
        pytest.param("set_ncpus", (2,),2,"ppn",id="set_ncpus"),
        pytest.param("set_hostlist", ("host_A",),"host_A","hostname",id="set_hostlist_str"),
        pytest.param("set_hostlist", (["host_A","host_B"],),"host_A,host_B","hostname",id="set_hostlist_list[str]"),
    ],
)
def test_update_env_initialized(function, value, flag, result):
    pbsScheduler = BatchSettings(scheduler=SchedulerType.PbsScheduler)
    getattr(pbsScheduler, function)(*value)
    assert pbsScheduler.scheduler_args[flag] == result
    
def test_create_pbs_batch():
    pbsScheduler = BatchSettings(scheduler=SchedulerType.PbsScheduler)
    pbsScheduler.set_nodes(1)
    pbsScheduler.set_walltime("10:00:00")
    pbsScheduler.set_queue("default")
    pbsScheduler.set_account("myproject")
    pbsScheduler.set_ncpus(10)
    args = pbsScheduler.format_batch_args()
    print(f"here: {args}")
    assert args == [
        "-l nodes=1:ncpus=10",
        "-l walltime=10:00:00",
        "-q default",
        "-A myproject",
    ]

# @pytest.mark.parametrize(
#     "method,params",
#     [
#         pytest.param("set_tasks", (3,), id="set_tasks"),
#         pytest.param("set_smts", ("smts",), id="set_smts"),
#         pytest.param("set_cpus_per_task", (2,), id="set_cpus_per_task"),
#         pytest.param("set_project", ("project",), id="set_project"),
#         pytest.param("set_partition", ("project",), id="set_partition"),
#     ],
# )
# def test_unimplimented_setters_throw_warning(caplog, method, params):
#     from smartsim.settings.base import logger

#     prev_prop = logger.propagate
#     logger.propagate = True

#     with caplog.at_level(logging.WARNING):
#         caplog.clear()
#         pbsScheduler = BatchSettings(scheduler=SchedulerType.PbsScheduler)
#         try:
#             getattr(pbsScheduler, method)(*params)
#         finally:
#             logger.propagate = prev_prop

#         for rec in caplog.records:
#             if (
#                 logging.WARNING <= rec.levelno < logging.ERROR
#                 and (method and "not supported" and "qsub") in rec.msg
#             ):
#                 break
#         else:
#             pytest.fail(
#                 (
#                     f"No message stating method `{method}` is not "
#                     "implemented at `warning` level"
#                 )
#             )