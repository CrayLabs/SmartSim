from smartsim.settingshold import LaunchSettings
from smartsim.settingshold.translators.launch.dragon import DragonArgTranslator
from smartsim.settingshold.launchCommand import LauncherType
import pytest
import logging

def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    dragonLauncher = LaunchSettings(launcher=LauncherType.DragonLauncher)
    assert dragonLauncher.launcher_str() == LauncherType.DragonLauncher.value

@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,),2,"nodes",id="set_nodes"),
        pytest.param("set_tasks_per_node", (2,),2,"tasks-per-node",id="set_tasks_per_node"),
    ],
)
def test_update_env_initialized(function, value, flag, result):
    dragonLauncher = LaunchSettings(launcher=LauncherType.DragonLauncher)
    assert dragonLauncher.launcher.value == LauncherType.DragonLauncher.value
    assert isinstance(dragonLauncher.arg_translator,DragonArgTranslator)
    getattr(dragonLauncher, function)(*value)
    assert dragonLauncher.launcher_args[flag] == result

@pytest.mark.parametrize(
    "method,params",
    [
        pytest.param("set_cpu_binding_type", ("bind",), id="set_cpu_binding_type"),
        pytest.param("set_task_map", ("task:map",), id="set_task_map"),
        pytest.param("set_hostlist", ("host_A",),id="set_hostlist_str"),
        pytest.param("set_hostlist", (["host_A","host_B"],),id="set_hostlist_list[str]"),
        pytest.param("set_hostlist_from_file", ("./path/to/hostfile",),id="set_hostlist_from_file"),
        pytest.param("set_excluded_hosts", ("host_A",),id="set_excluded_hosts_str"),
        pytest.param("set_excluded_hosts", (["host_A","host_B"],),id="set_excluded_hosts_list[str]"),
        pytest.param("set_cpu_bindings", (4,),id="set_cpu_bindings"),
        pytest.param("set_cpu_bindings", ([4,4],),id="set_cpu_bindings_list[str]"),
        pytest.param("set_verbose_launch", (True,),id="set_verbose_launch"),
        pytest.param("set_quiet_launch", (True,),id="set_quiet_launch"),
        pytest.param("set_executable_broadcast", ("/tmp/some/path",),id="set_broadcast"),
        pytest.param("set_walltime", ("10:00:00",),id="set_walltime"),
        pytest.param("set_node_feature", ("P100",),id="set_node_feature"),
        pytest.param("set_memory_per_node", ("1000",),id="set_memory_per_node"),
        pytest.param("set_tasks", (2,),id="set_tasks"),
        pytest.param("set_binding", ("bind",),id="set_tasks"),
        pytest.param("format_comma_sep_env_vars", (), id="format_comma_sep_env_vars"),
        pytest.param("format_launcher_args", (), id="format_launcher_args"),
        pytest.param("format_env_vars", (), id="format_env_vars"),
    ],
)
def test_unimplimented_setters_throw_warning(caplog, method, params):
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        dragonLauncher = LaunchSettings(launcher=LauncherType.DragonLauncher)
        try:
            getattr(dragonLauncher, method)(*params)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and (method and "not supported" and "dragon") in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )