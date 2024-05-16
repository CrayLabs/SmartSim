from smartsim.settingshold import BatchSettings
from smartsim.settingshold.translators.batch.pbs import QsubBatchArgTranslator
import pytest
import logging
    
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
    pbsScheduler = BatchSettings(scheduler="qsub")
    getattr(pbsScheduler, function)(*value)
    assert pbsScheduler.scheduler_args[flag] == result
    
def test_create_pbs_batch():
    pbsScheduler = BatchSettings(scheduler="qsub")
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