from smartsim.settingshold import LaunchSettings
from smartsim.settingshold.translators.launch.lsf import JsrunArgTranslator
from smartsim.settingshold.launchCommand import LauncherType
import pytest
import logging

def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    lsfLauncher = LaunchSettings(launcher=LauncherType.LsfLauncher)
    assert lsfLauncher.launcher_str() == LauncherType.LsfLauncher.value

def test_set_reserved_launcher_args():
    """Ensure launcher_str returns appropriate value"""
    lsfLauncher = LaunchSettings(launcher=LauncherType.LsfLauncher)
    assert lsfLauncher._reserved_launch_args == {"chdir", "h"}

@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_tasks", (2,),2,"np",id="set_tasks"),
        pytest.param("set_binding", ("packed:21",),"packed:21","bind",id="set_binding"),
    ],
)
def test_update_env_initialized(function, value, flag, result):
    lsfLauncher = LaunchSettings(launcher=LauncherType.LsfLauncher)
    assert lsfLauncher.launcher.value == LauncherType.LsfLauncher.value
    assert isinstance(lsfLauncher.arg_translator,JsrunArgTranslator)
    getattr(lsfLauncher, function)(*value)
    assert lsfLauncher.launcher_args[flag] == result

def test_format_env_vars():
    env_vars = {"OMP_NUM_THREADS": None, "LOGGING": "verbose"}
    lsfLauncher = LaunchSettings(launcher=LauncherType.LsfLauncher, env_vars=env_vars)
    assert lsfLauncher.launcher.value == LauncherType.LsfLauncher.value
    assert isinstance(lsfLauncher.arg_translator,JsrunArgTranslator)
    formatted = lsfLauncher.format_env_vars()
    assert formatted == ["-E", "OMP_NUM_THREADS", "-E", "LOGGING=verbose"]

def test_launch_args():
    """Test the possible user overrides through run_args"""
    launch_args = {
        "latency_priority": "gpu-gpu",
        "immediate": None,
        "d": "packed",  # test single letter variables
        "nrs": 10,
        "np": 100,
    }
    lsfLauncher = LaunchSettings(launcher=LauncherType.LsfLauncher, launcher_args=launch_args)
    assert lsfLauncher.launcher.value == "jsrun"
    formatted = lsfLauncher.format_launcher_args()
    result = [
        "--latency_priority=gpu-gpu",
        "--immediate",
        "-d",
        "packed",
        "--nrs=10",
        "--np=100",
    ]
    assert formatted == result

@pytest.mark.parametrize(
    "method,params",
    [
        pytest.param("set_cpu_binding_type", ("bind",), id="set_cpu_binding_type"),
        pytest.param("set_task_map", ("task:map",), id="set_task_map"),
        pytest.param("set_nodes", (2,), id="set_nodes"),
        pytest.param("set_hostlist", ("host_A",),id="set_hostlist_str"),
        pytest.param("set_hostlist", (["host_A","host_B"],),id="set_hostlist_list[str]"),
        pytest.param("set_hostlist_from_file", ("./path/to/hostfile",),id="set_hostlist_from_file"),
        pytest.param("set_excluded_hosts", ("host_A",),id="set_excluded_hosts_str"),
        pytest.param("set_excluded_hosts", (["host_A","host_B"],),id="set_excluded_hosts_list[str]"),
        pytest.param("set_cpu_bindings", (4,),id="set_cpu_bindings"),
        pytest.param("set_cpu_bindings", ([4,4],),id="set_cpu_bindings_list[str]"),
        pytest.param("set_cpus_per_task", (2,),id="set_cpus_per_task"),
        pytest.param("set_verbose_launch", (True,),id="set_verbose_launch"),
        pytest.param("set_quiet_launch", (True,),id="set_quiet_launch"),
        pytest.param("set_executable_broadcast", ("/tmp/some/path",),id="set_broadcast"),
        pytest.param("set_walltime", ("10:00:00",),id="set_walltime"),
        pytest.param("set_node_feature", ("P100",),id="set_node_feature"),
        pytest.param("set_memory_per_node", ("1000",),id="set_memory_per_node"),
        pytest.param("set_cpu_binding_type", ("bind",), id="set_cpu_binding_type"),
        pytest.param("set_executable_broadcast", ("broad",),id="set_executable_broadcast"),
        pytest.param("format_comma_sep_env_vars", (), id="format_comma_sep_env_vars"),
    ],
)
def test_unimplimented_setters_throw_warning(caplog, method, params):
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        lsfLauncher = LaunchSettings(launcher=LauncherType.LsfLauncher)
        try:
            getattr(lsfLauncher, method)(*params)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and (method and "not supported" and "jsrun") in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )