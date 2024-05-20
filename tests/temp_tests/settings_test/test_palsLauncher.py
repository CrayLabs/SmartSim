from smartsim.settingshold import LaunchSettings
from smartsim.settingshold.translators.launch.pals import PalsMpiexecArgTranslator
from smartsim.settingshold.launchCommand import LauncherType
import pytest
import logging

def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    palsLauncher = LaunchSettings(launcher=LauncherType.PalsLauncher)
    assert palsLauncher.launcher_str() == LauncherType.PalsLauncher.value

def test_set_reserved_launcher_args():
    """Ensure launcher_str returns appropriate value"""
    palsLauncher = LaunchSettings(launcher=LauncherType.PalsLauncher)
    assert palsLauncher._reserved_launch_args == set()

@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_cpu_binding_type", ("bind",),"bind","bind-to",id="set_cpu_binding_type"),
        pytest.param("set_tasks", (2,),2,"np",id="set_tasks"),
        pytest.param("set_tasks_per_node", (2,),2,"ppn",id="set_tasks_per_node"),
        pytest.param("set_hostlist", ("host_A",),"host_A","hosts",id="set_hostlist_str"),
        pytest.param("set_hostlist", (["host_A","host_B"],),"host_A,host_B","hosts",id="set_hostlist_list[str]"),
        pytest.param("set_executable_broadcast", ("broadcast",),"broadcast","transfer",id="set_executable_broadcast"),
    ],
)
def test_pals_class_methods(function, value, flag, result):
    palsLauncher = LaunchSettings(launcher=LauncherType.PalsLauncher)
    getattr(palsLauncher, function)(*value)
    assert palsLauncher.launcher == LauncherType.PalsLauncher
    assert isinstance(palsLauncher.arg_translator,PalsMpiexecArgTranslator)
    assert palsLauncher.launcher_args[flag] == result
    assert palsLauncher.format_launcher_args() == ["--" + flag, str(result)]

def test_format_env_vars():
    env_vars = {"FOO_VERSION": "3.14", "PATH": None, "LD_LIBRARY_PATH": None}
    palsLauncher = LaunchSettings(launcher=LauncherType.PalsLauncher, env_vars=env_vars)
    formatted = " ".join(palsLauncher.format_env_vars())
    expected = "--env FOO_VERSION=3.14 --envlist PATH,LD_LIBRARY_PATH"
    assert formatted == expected

def test_invalid_hostlist_format():
    """Test invalid hostlist formats"""
    palsLauncher = LaunchSettings(launcher=LauncherType.PalsLauncher)
    with pytest.raises(TypeError):
        palsLauncher.set_hostlist(["test",5])
    with pytest.raises(TypeError):
        palsLauncher.set_hostlist([5])
    with pytest.raises(TypeError):
        palsLauncher.set_hostlist(5)

@pytest.mark.parametrize(
    "method,params",
    [
        pytest.param("set_cpu_bindings", ("bind",), id="set_cpu_bindings"),
        pytest.param("set_binding", ("bind",), id="set_binding"),
        pytest.param("set_nodes", (2,), id="set_nodes"),
        pytest.param("set_task_map", ("task:map",), id="set_task_map"),
        pytest.param("set_cpus_per_task", ("task:map",), id="set_cpus_per_task"),
        pytest.param("set_quiet_launch", ("task:map",), id="set_quiet_launch"),
        pytest.param("set_walltime", ("task:map",), id="set_walltime"),
        pytest.param("set_node_feature", ("P100",),id="set_node_feature"),
        pytest.param("set_hostlist_from_file", ("./file",),id="set_hostlist_from_file"),
        pytest.param("set_excluded_hosts", ("./file",),id="set_excluded_hosts"),
        pytest.param("set_memory_per_node", ("8000",),id="set_memory_per_node"),
        pytest.param("set_verbose_launch", (True,),id="set_verbose_launch"),
        pytest.param("set_quiet_launch", (True,),id="set_quiet_launch"),
        pytest.param("format_comma_sep_env_vars", (), id="format_comma_sep_env_vars"),
        pytest.param("set_het_group", ([1,2,3,4],), id="set_het_group"),
    ],
)
def test_unimplimented_setters_throw_warning(caplog, method, params):
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        palsLauncher = LaunchSettings(launcher=LauncherType.PalsLauncher)
        try:
            getattr(palsLauncher, method)(*params)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and ("not supported" and "pals") in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )