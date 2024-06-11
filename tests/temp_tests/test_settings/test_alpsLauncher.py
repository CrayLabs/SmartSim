import pytest

from smartsim.settings import LaunchSettings
from smartsim.settings.builders.launch.alps import AprunArgBuilder
from smartsim.settings.launchCommand import LauncherType


def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    alpsLauncher = LaunchSettings(launcher=LauncherType.Alps)
    assert alpsLauncher.launch_args.launcher_str() == LauncherType.Alps.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param(
            "set_cpus_per_task", (4,), "4", "cpus-per-pe", id="set_cpus_per_task"
        ),
        pytest.param("set_tasks", (4,), "4", "pes", id="set_tasks"),
        pytest.param(
            "set_tasks_per_node", (4,), "4", "pes-per-node", id="set_tasks_per_node"
        ),
        pytest.param(
            "set_hostlist", ("host_A",), "host_A", "node-list", id="set_hostlist_str"
        ),
        pytest.param(
            "set_hostlist",
            (["host_A", "host_B"],),
            "host_A,host_B",
            "node-list",
            id="set_hostlist_list[str]",
        ),
        pytest.param(
            "set_hostlist_from_file",
            ("./path/to/hostfile",),
            "./path/to/hostfile",
            "node-list-file",
            id="set_hostlist_from_file",
        ),
        pytest.param(
            "set_excluded_hosts",
            ("host_A",),
            "host_A",
            "exclude-node-list",
            id="set_excluded_hosts_str",
        ),
        pytest.param(
            "set_excluded_hosts",
            (["host_A", "host_B"],),
            "host_A,host_B",
            "exclude-node-list",
            id="set_excluded_hosts_list[str]",
        ),
        pytest.param(
            "set_cpu_bindings", (4,), "4", "cpu-binding", id="set_cpu_bindings"
        ),
        pytest.param(
            "set_cpu_bindings",
            ([4, 4],),
            "4,4",
            "cpu-binding",
            id="set_cpu_bindings_list[str]",
        ),
        pytest.param(
            "set_memory_per_node",
            (8000,),
            "8000",
            "memory-per-pe",
            id="set_memory_per_node",
        ),
        pytest.param(
            "set_walltime",
            ("10:00:00",),
            "10:00:00",
            "cpu-time-limit",
            id="set_walltime",
        ),
        pytest.param(
            "set_verbose_launch", (True,), "7", "debug", id="set_verbose_launch"
        ),
        pytest.param("set_quiet_launch", (True,), None, "quiet", id="set_quiet_launch"),
    ],
)
def test_alps_class_methods(function, value, flag, result):
    alpsLauncher = LaunchSettings(launcher=LauncherType.Alps)
    assert isinstance(alpsLauncher._arg_builder, AprunArgBuilder)
    getattr(alpsLauncher.launch_args, function)(*value)
    assert alpsLauncher.launch_args._launch_args[flag] == result


def test_set_verbose_launch():
    alpsLauncher = LaunchSettings(launcher=LauncherType.Alps)
    assert isinstance(alpsLauncher._arg_builder, AprunArgBuilder)
    alpsLauncher.launch_args.set_verbose_launch(True)
    assert alpsLauncher.launch_args._launch_args == {"debug": "7"}
    alpsLauncher.launch_args.set_verbose_launch(False)
    assert alpsLauncher.launch_args._launch_args == {}


def test_set_quiet_launch():
    aprunLauncher = LaunchSettings(launcher=LauncherType.Alps)
    assert isinstance(aprunLauncher._arg_builder, AprunArgBuilder)
    aprunLauncher.launch_args.set_quiet_launch(True)
    assert aprunLauncher.launch_args._launch_args == {"quiet": None}
    aprunLauncher.launch_args.set_quiet_launch(False)
    assert aprunLauncher.launch_args._launch_args == {}


def test_format_env_vars():
    env_vars = {"OMP_NUM_THREADS": "20", "LOGGING": "verbose"}
    aprunLauncher = LaunchSettings(launcher=LauncherType.Alps, env_vars=env_vars)
    assert isinstance(aprunLauncher._arg_builder, AprunArgBuilder)
    aprunLauncher.update_env({"OMP_NUM_THREADS": "10"})
    formatted = aprunLauncher.format_env_vars()
    result = ["-e", "OMP_NUM_THREADS=10", "-e", "LOGGING=verbose"]
    assert formatted == result


def test_aprun_settings():
    aprunLauncher = LaunchSettings(launcher=LauncherType.Alps)
    aprunLauncher.launch_args.set_cpus_per_task(2)
    aprunLauncher.launch_args.set_tasks(100)
    aprunLauncher.launch_args.set_tasks_per_node(20)
    formatted = aprunLauncher.format_launch_args()
    result = ["--cpus-per-pe=2", "--pes=100", "--pes-per-node=20"]
    assert formatted == result


def test_invalid_hostlist_format():
    """Test invalid hostlist formats"""
    alpsLauncher = LaunchSettings(launcher=LauncherType.Alps)
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_hostlist(["test", 5])
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_hostlist([5])
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_hostlist(5)


def test_invalid_exclude_hostlist_format():
    """Test invalid hostlist formats"""
    alpsLauncher = LaunchSettings(launcher=LauncherType.Alps)
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_excluded_hosts(["test", 5])
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_excluded_hosts([5])
    with pytest.raises(TypeError):
        alpsLauncher.launch_args.set_excluded_hosts(5)
