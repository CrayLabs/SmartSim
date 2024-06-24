import pytest

from smartsim.settings import LaunchSettings
from smartsim.settings.builders.launch.pals import PalsMpiexecArgBuilder
from smartsim.settings.launchCommand import LauncherType


def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    ls = LaunchSettings(launcher=LauncherType.Pals)
    assert ls.launch_args.launcher_str() == LauncherType.Pals.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param(
            "set_cpu_binding_type",
            ("bind",),
            "bind",
            "bind-to",
            id="set_cpu_binding_type",
        ),
        pytest.param("set_tasks", (2,), "2", "np", id="set_tasks"),
        pytest.param("set_tasks_per_node", (2,), "2", "ppn", id="set_tasks_per_node"),
        pytest.param(
            "set_hostlist", ("host_A",), "host_A", "hosts", id="set_hostlist_str"
        ),
        pytest.param(
            "set_hostlist",
            (["host_A", "host_B"],),
            "host_A,host_B",
            "hosts",
            id="set_hostlist_list[str]",
        ),
        pytest.param(
            "set_executable_broadcast",
            ("broadcast",),
            "broadcast",
            "transfer",
            id="set_executable_broadcast",
        ),
    ],
)
def test_pals_class_methods(function, value, flag, result):
    palsLauncher = LaunchSettings(launcher=LauncherType.Pals)
    assert isinstance(palsLauncher.launch_args, PalsMpiexecArgBuilder)
    getattr(palsLauncher.launch_args, function)(*value)
    assert palsLauncher.launch_args._launch_args[flag] == result
    assert palsLauncher.format_launch_args() == ["--" + flag, str(result)]


def test_format_env_vars():
    env_vars = {"FOO_VERSION": "3.14", "PATH": None, "LD_LIBRARY_PATH": None}
    palsLauncher = LaunchSettings(launcher=LauncherType.Pals, env_vars=env_vars)
    formatted = " ".join(palsLauncher.format_env_vars())
    expected = "--env FOO_VERSION=3.14 --envlist PATH,LD_LIBRARY_PATH"
    assert formatted == expected


def test_invalid_hostlist_format():
    """Test invalid hostlist formats"""
    palsLauncher = LaunchSettings(launcher=LauncherType.Pals)
    with pytest.raises(TypeError):
        palsLauncher.launch_args.set_hostlist(["test", 5])
    with pytest.raises(TypeError):
        palsLauncher.launch_args.set_hostlist([5])
    with pytest.raises(TypeError):
        palsLauncher.launch_args.set_hostlist(5)
