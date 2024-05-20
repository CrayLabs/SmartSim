from smartsim.settingshold import LaunchSettings
from smartsim.settingshold.translators.launch.local import LocalArgTranslator
from smartsim.settingshold.launchCommand import LauncherType
import pytest
import logging

def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    localLauncher = LaunchSettings(launcher=LauncherType.LocalLauncher)
    assert localLauncher.launcher_str() == LauncherType.LocalLauncher.value

def test_set_reserved_launcher_args():
    """Ensure launcher_str returns appropriate value"""
    localLauncher = LaunchSettings(launcher=LauncherType.LocalLauncher)
    assert localLauncher._reserved_launch_args == set()

def test_launch_args_input_mutation():
    # Tests that the run args passed in are not modified after initialization
    key0, key1, key2 = "arg0", "arg1", "arg2"
    val0, val1, val2 = "val0", "val1", "val2"

    default_launcher_args = {
        key0: val0,
        key1: val1,
        key2: val2,
    }
    localLauncher = LaunchSettings(launcher=LauncherType.LocalLauncher, launcher_args=default_launcher_args)

    # Confirm initial values are set
    assert localLauncher.launcher_args[key0] == val0
    assert localLauncher.launcher_args[key1] == val1
    assert localLauncher.launcher_args[key2] == val2

    # Update our common run arguments
    val2_upd = f"not-{val2}"
    default_launcher_args[key2] = val2_upd

    # Confirm previously created run settings are not changed
    assert localLauncher.launcher_args[key2] == val2

@pytest.mark.parametrize(
    "env_vars",
    [
        pytest.param({}, id="no env vars"),
        pytest.param({"env1": "abc"}, id="normal var"),
        pytest.param({"env1": "abc,def"}, id="compound var"),
        pytest.param({"env1": "xyz", "env2": "pqr"}, id="multiple env vars"),
    ],
)
def test_update_env(env_vars):
    """Ensure non-initialized env vars update correctly"""
    localLauncher = LaunchSettings(launcher=LauncherType.LocalLauncher)
    localLauncher.update_env(env_vars)

    assert len(localLauncher.env_vars) == len(env_vars.keys())

def test_format_launcher_args():
    localLauncher = LaunchSettings(launcher=LauncherType.LocalLauncher, launcher_args={"-np": 2})
    launcher_args = localLauncher.format_launcher_args()
    assert launcher_args == ["-np", "2"]

@pytest.mark.parametrize(
    "env_vars",
    [
        pytest.param({"env1": None}, id="null value not allowed"),
        pytest.param({"env1": {"abc"}}, id="set value not allowed"),
        pytest.param({"env1": {"abc": "def"}}, id="dict value not allowed"),
    ],
)
def test_update_env_null_valued(env_vars):
    """Ensure validation of env var in update"""
    orig_env = {}

    with pytest.raises(TypeError) as ex:
        localLauncher = LaunchSettings(launcher=LauncherType.LocalLauncher, env_vars=orig_env)
        localLauncher.update_env(env_vars)

@pytest.mark.parametrize(
    "env_vars",
    [
        pytest.param({}, id="no env vars"),
        pytest.param({"env1": "abc"}, id="normal var"),
        pytest.param({"env1": "abc,def"}, id="compound var"),
        pytest.param({"env1": "xyz", "env2": "pqr"}, id="multiple env vars"),
    ],
)
def test_update_env_initialized(env_vars):
    """Ensure update of initialized env vars does not overwrite"""
    orig_env = {"key": "value"}
    localLauncher = LaunchSettings(launcher=LauncherType.LocalLauncher, env_vars=orig_env)
    localLauncher.update_env(env_vars)

    combined_keys = {k for k in env_vars.keys()}
    combined_keys.update(k for k in orig_env.keys())

    assert len(localLauncher.env_vars) == len(combined_keys)
    assert {k for k in localLauncher.env_vars.keys()} == combined_keys

def test_format_env_vars():
    env_vars={
            "A": "a",
            "B": None,
            "C": "",
            "D": "12",
        }
    localLauncher = LaunchSettings(launcher=LauncherType.LocalLauncher, env_vars=env_vars)
    assert localLauncher.launcher.value == LauncherType.LocalLauncher.value
    assert isinstance(localLauncher.arg_translator, LocalArgTranslator)
    assert localLauncher.format_env_vars() == ["A=a", "B=", "C=", "D=12"]

@pytest.mark.parametrize(
    "method,params",
    [
        pytest.param("set_nodes", (2,), id="set_nodes"),
        pytest.param("set_tasks", (2,), id="set_tasks"),
        pytest.param("set_tasks_per_node", (3,), id="set_tasks_per_node"),
        pytest.param("set_task_map", (3,), id="set_task_map"),
        pytest.param("set_cpus_per_task", (4,), id="set_cpus_per_task"),
        pytest.param("set_hostlist", ("hostlist",), id="set_hostlist"),
        pytest.param("set_node_feature", ("P100",), id="set_node_feature"),
        pytest.param(
            "set_hostlist_from_file", ("~/hostfile",), id="set_hostlist_from_file"
        ),
        pytest.param("set_excluded_hosts", ("hostlist",), id="set_excluded_hosts"),
        pytest.param("set_cpu_bindings", ([1, 2, 3],), id="set_cpu_bindings"),
        pytest.param("set_memory_per_node", (16_000,), id="set_memory_per_node"),
        pytest.param("set_verbose_launch", (False,), id="set_verbose_launch"),
        pytest.param("set_quiet_launch", (True,), id="set_quiet_launch"),
        pytest.param("set_walltime", ("00:55:00",), id="set_walltime"),
        pytest.param("set_executable_broadcast", ("broad",), id="set_executable_broadcast"),
        pytest.param("set_binding", ("packed:21",), id="set_binding"),
        pytest.param("set_cpu_binding_type", ("bind",), id="set_cpu_binding_type"),
        pytest.param("format_comma_sep_env_vars", (), id="format_comma_sep_env_vars"),
    ],
)
def test_unimplimented_setters_throw_warning(caplog, method, params):
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        localLauncher = LaunchSettings(launcher=LauncherType.LocalLauncher)
        try:
            getattr(localLauncher, method)(*params)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and ("method" and "not supported" and "local") in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )