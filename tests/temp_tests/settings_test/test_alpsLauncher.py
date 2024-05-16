from smartsim.settingshold import LaunchSettings
from smartsim.settingshold.translators.launch.alps import AprunArgTranslator
import pytest
    
@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_cpus_per_task", (4,),4,"--cpus-per-pe",id="set_cpus_per_task"),
        pytest.param("set_tasks", (4,),4,"--pes",id="set_tasks"),
        pytest.param("set_tasks_per_node", (4,),4,"--pes-per-node",id="set_tasks_per_node"),
        pytest.param("set_hostlist", ("host_A",),"host_A","--node-list",id="set_hostlist_str"),
        pytest.param("set_hostlist", (["host_A","host_B"],),"host_A,host_B","--node-list",id="set_hostlist_list[str]"),
        pytest.param("set_hostlist_from_file", ("./path/to/hostfile",),"./path/to/hostfile","--node-list-file",id="set_hostlist_from_file"),
        pytest.param("set_excluded_hosts", ("host_A",),"host_A","--exclude-node-list",id="set_excluded_hosts_str"),
        pytest.param("set_excluded_hosts", (["host_A","host_B"],),"host_A,host_B","--exclude-node-list",id="set_excluded_hosts_list[str]"),
        pytest.param("set_cpu_bindings", (4,),"4","--cpu-binding",id="set_cpu_bindings"),
        pytest.param("set_cpu_bindings", ([4,4],),"4,4","--cpu-binding",id="set_cpu_bindings_list[str]"),
        pytest.param("set_memory_per_node", (8000,),8000,"--memory-per-pe",id="set_memory_per_node"),
        pytest.param("set_walltime", ("10:00:00",),"10:00:00","--cpu-time-limit",id="set_walltime"),
    ],
)
def test_update_env_initialized(function, value, flag, result):
    alpsLauncher = LaunchSettings(launcher="aprun")
    assert alpsLauncher.launcher == "aprun"
    assert isinstance(alpsLauncher.arg_translator,AprunArgTranslator)
    getattr(alpsLauncher, function)(*value)
    assert alpsLauncher.launcher_args[flag] == result

def test_set_verbose_launch():
    slurmLauncher = LaunchSettings(launcher="aprun")
    assert slurmLauncher.launcher == "aprun"
    assert isinstance(slurmLauncher.arg_translator,AprunArgTranslator)
    slurmLauncher.set_verbose_launch(True)
    assert slurmLauncher.launcher_args == {'--debug': 7}
    slurmLauncher.set_verbose_launch(False)
    assert slurmLauncher.launcher_args == {}

def test_set_quiet_launch():
    slurmLauncher = LaunchSettings(launcher="aprun")
    assert slurmLauncher.launcher == "aprun"
    assert isinstance(slurmLauncher.arg_translator,AprunArgTranslator)
    slurmLauncher.set_quiet_launch(True)
    assert slurmLauncher.launcher_args == {'--quiet': None}
    slurmLauncher.set_quiet_launch(False)
    assert slurmLauncher.launcher_args == {}
    
def test_format_env_vars():
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose"}
    slurmLauncher = LaunchSettings(launcher="aprun", env_vars=env_vars)
    assert slurmLauncher.launcher == "aprun"
    formatted = slurmLauncher.format_env_vars()
    assert "OMP_NUM_THREADS=20" in formatted
    assert "LOGGING=verbose" in formatted
    

@pytest.mark.parametrize(
    "method,params",
    [
        pytest.param("set_nodes", (2,),2,"nodes",id="set_nodes"),
        pytest.param("set_executable_broadcast", ("/tmp/some/path",),"/tmp/some/path","bcast",id="set_executable_broadcast"),
        pytest.param("set_node_feature", ("P100",),"P100","C",id="set_node_feature"),
        pytest.param("set_cpu_binding_type", ("bind",), id="set_cpu_binding_type"),
        pytest.param("set_task_map", ("task:map",), id="set_task_map"),
        pytest.param("set_binding", ("bind",), id="set_binding"),
    ],
)
def test_unimplimented_methods_throw_warning(caplog, method, params):
    """Test methods not implemented throw warnings"""
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        slurmLauncher = LaunchSettings(launcher="slurm")
        try:
            getattr(slurmLauncher, method)(*params)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and ("not supported" and "slurm") in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )