import pytest
from smartsim.settingshold import LaunchSettings
from smartsim.settingshold.translators.launch.alps import AprunArgTranslator
from smartsim.settingshold.launchCommand import LauncherType
import logging

def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    alpsLauncher = LaunchSettings(launcher=LauncherType.AlpsLauncher)
    assert alpsLauncher.launcher_str() == LauncherType.AlpsLauncher.value

def test_set_reserved_launcher_args():
    """Ensure launcher_str returns appropriate value"""
    alpsLauncher = LaunchSettings(launcher=LauncherType.AlpsLauncher)
    assert alpsLauncher._reserved_launch_args == set()

@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_cpus_per_task", (4,),4,"cpus-per-pe",id="set_cpus_per_task"),
        pytest.param("set_tasks", (4,),4,"pes",id="set_tasks"),
        pytest.param("set_tasks_per_node", (4,),4,"pes-per-node",id="set_tasks_per_node"),
        pytest.param("set_hostlist", ("host_A",),"host_A","node-list",id="set_hostlist_str"),
        pytest.param("set_hostlist", (["host_A","host_B"],),"host_A,host_B","node-list",id="set_hostlist_list[str]"),
        pytest.param("set_hostlist_from_file", ("./path/to/hostfile",),"./path/to/hostfile","node-list-file",id="set_hostlist_from_file"),
        pytest.param("set_excluded_hosts", ("host_A",),"host_A","exclude-node-list",id="set_excluded_hosts_str"),
        pytest.param("set_excluded_hosts", (["host_A","host_B"],),"host_A,host_B","exclude-node-list",id="set_excluded_hosts_list[str]"),
        pytest.param("set_cpu_bindings", (4,),"4","cpu-binding",id="set_cpu_bindings"),
        pytest.param("set_cpu_bindings", ([4,4],),"4,4","cpu-binding",id="set_cpu_bindings_list[str]"),
        pytest.param("set_memory_per_node", (8000,),"8000","memory-per-pe",id="set_memory_per_node"),
        pytest.param("set_walltime", ("10:00:00",),"10:00:00","cpu-time-limit",id="set_walltime"),
        pytest.param("set_verbose_launch", (True,),7,"debug",id="set_verbose_launch"),
        pytest.param("set_quiet_launch", (True,),None,"quiet",id="set_quiet_launch"),
    ],
)
def test_alps_class_methods(function, value, flag, result):
    alpsLauncher = LaunchSettings(launcher=LauncherType.AlpsLauncher)
    assert alpsLauncher.launcher.value == LauncherType.AlpsLauncher.value
    assert isinstance(alpsLauncher.arg_translator,AprunArgTranslator)
    getattr(alpsLauncher, function)(*value)
    assert alpsLauncher.launcher_args[flag] == result

def test_set_verbose_launch():
    alpsLauncher = LaunchSettings(launcher=LauncherType.AlpsLauncher)
    assert alpsLauncher.launcher.value == LauncherType.AlpsLauncher.value
    assert isinstance(alpsLauncher.arg_translator,AprunArgTranslator)
    alpsLauncher.set_verbose_launch(True)
    assert alpsLauncher.launcher_args == {'debug': 7}
    alpsLauncher.set_verbose_launch(False)
    assert alpsLauncher.launcher_args == {}

def test_set_quiet_launch():
    aprunLauncher = LaunchSettings(launcher=LauncherType.AlpsLauncher)
    assert aprunLauncher.launcher.value == LauncherType.AlpsLauncher.value
    assert isinstance(aprunLauncher.arg_translator,AprunArgTranslator)
    aprunLauncher.set_quiet_launch(True)
    assert aprunLauncher.launcher_args == {'quiet': None}
    aprunLauncher.set_quiet_launch(False)
    assert aprunLauncher.launcher_args == {}
    
def test_format_env_vars():
    env_vars = {"OMP_NUM_THREADS": "20", "LOGGING": "verbose"}
    aprunLauncher = LaunchSettings(launcher=LauncherType.AlpsLauncher, env_vars=env_vars)
    assert aprunLauncher.launcher.value == LauncherType.AlpsLauncher.value
    aprunLauncher.update_env({"OMP_NUM_THREADS": "10"})
    formatted = aprunLauncher.format_env_vars()
    result = ["-e", "OMP_NUM_THREADS=10", "-e", "LOGGING=verbose"]
    assert formatted == result

def test_aprun_settings():
    aprunLauncher = LaunchSettings(launcher=LauncherType.AlpsLauncher)
    aprunLauncher.set_cpus_per_task(2)
    aprunLauncher.set_tasks(100)
    aprunLauncher.set_tasks_per_node(20)
    formatted = aprunLauncher.format_launcher_args()
    result = ["--cpus-per-pe=2", "--pes=100", "--pes-per-node=20"]
    assert formatted == result

def test_invalid_hostlist_format():
    """Test invalid hostlist formats"""
    alpsLauncher = LaunchSettings(launcher=LauncherType.AlpsLauncher)
    with pytest.raises(TypeError):
        alpsLauncher.set_hostlist(["test",5])
    with pytest.raises(TypeError):
        alpsLauncher.set_hostlist([5])
    with pytest.raises(TypeError):
        alpsLauncher.set_hostlist(5)

def test_invalid_exclude_hostlist_format():
    """Test invalid hostlist formats"""
    alpsLauncher = LaunchSettings(launcher=LauncherType.AlpsLauncher)
    with pytest.raises(TypeError):
        alpsLauncher.set_excluded_hosts(["test",5])
    with pytest.raises(TypeError):
        alpsLauncher.set_excluded_hosts([5])
    with pytest.raises(TypeError):
        alpsLauncher.set_excluded_hosts(5)

@pytest.mark.parametrize(
    "method,params",
    [
        pytest.param("set_nodes", (2,),id="set_nodes"),
        pytest.param("set_executable_broadcast", ("/tmp/some/path",),id="set_broadcast"),
        pytest.param("set_node_feature", ("P100",),id="set_node_feature"),
        pytest.param("set_cpu_binding_type", ("bind",), id="set_cpu_binding_type"),
        pytest.param("set_task_map", ("task:map",), id="set_task_map"),
        pytest.param("set_binding", ("bind",), id="set_binding"),
        pytest.param("format_comma_sep_env_vars", (), id="format_comma_sep_env_vars"),
        pytest.param("set_het_group", ([1,2,3,4],), id="set_het_group"),
    ],
)
def test_unimplimented_methods_throw_warning(caplog, method, params):
    """Test methods not implemented throw warnings"""
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        alpsLauncher = LaunchSettings(launcher=LauncherType.AlpsLauncher)
        try:
            getattr(alpsLauncher, method)(*params)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and (method and "not supported" and "aprun") in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )