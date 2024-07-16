import itertools

import pytest

from smartsim.settings import LaunchSettings
from smartsim.settings.builders.launch.mpi import (
    MpiArgBuilder,
    MpiexecArgBuilder,
    OrteArgBuilder,
)
from smartsim.settings.launchCommand import LauncherType

pytestmark = pytest.mark.group_a


@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.Mpirun, id="launcher_str_mpirun"),
        pytest.param(LauncherType.Mpiexec, id="launcher_str_mpiexec"),
        pytest.param(LauncherType.Orterun, id="launcher_str_orterun"),
    ],
)
def test_launcher_str(launcher):
    """Ensure launcher_str returns appropriate value"""
    ls = LaunchSettings(launcher=launcher)
    assert ls.launch_args.launcher_str() == launcher.value


@pytest.mark.parametrize(
    "l,function,value,result,flag",
    [
        # Use OpenMPI style settigs for all launchers
        *itertools.chain.from_iterable(
            (
                (
                    pytest.param(
                        l, "set_walltime", ("100",), "100", "timeout", id="set_walltime"
                    ),
                    pytest.param(
                        l,
                        "set_task_map",
                        ("taskmap",),
                        "taskmap",
                        "map-by",
                        id="set_task_map",
                    ),
                    pytest.param(
                        l,
                        "set_cpus_per_task",
                        (2,),
                        "2",
                        "cpus-per-proc",
                        id="set_cpus_per_task",
                    ),
                    pytest.param(
                        l,
                        "set_cpu_binding_type",
                        ("4",),
                        "4",
                        "bind-to",
                        id="set_cpu_binding_type",
                    ),
                    pytest.param(
                        l,
                        "set_tasks_per_node",
                        (4,),
                        "4",
                        "npernode",
                        id="set_tasks_per_node",
                    ),
                    pytest.param(l, "set_tasks", (4,), "4", "n", id="set_tasks"),
                    pytest.param(
                        l,
                        "set_executable_broadcast",
                        ("broadcast",),
                        "broadcast",
                        "preload-binary",
                        id="set_executable_broadcast",
                    ),
                    pytest.param(
                        l,
                        "set_hostlist",
                        ("host_A",),
                        "host_A",
                        "host",
                        id="set_hostlist_str",
                    ),
                    pytest.param(
                        l,
                        "set_hostlist",
                        (["host_A", "host_B"],),
                        "host_A,host_B",
                        "host",
                        id="set_hostlist_list[str]",
                    ),
                    pytest.param(
                        l,
                        "set_hostlist_from_file",
                        ("./path/to/hostfile",),
                        "./path/to/hostfile",
                        "hostfile",
                        id="set_hostlist_from_file",
                    ),
                )
                for l in (
                    [LauncherType.Mpirun, MpiArgBuilder],
                    [LauncherType.Mpiexec, MpiexecArgBuilder],
                    [LauncherType.Orterun, OrteArgBuilder],
                )
            )
        )
    ],
)
def test_mpi_class_methods(l, function, value, flag, result):
    mpiSettings = LaunchSettings(launcher=l[0])
    assert isinstance(mpiSettings._arg_builder, l[1])
    getattr(mpiSettings.launch_args, function)(*value)
    assert mpiSettings.launch_args._launch_args[flag] == result


@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.Mpirun, id="format_env_mpirun"),
        pytest.param(LauncherType.Mpiexec, id="format_env_mpiexec"),
        pytest.param(LauncherType.Orterun, id="format_env_orterun"),
    ],
)
def test_format_env_vars(launcher):
    env_vars = {"OMP_NUM_THREADS": "20", "LOGGING": "verbose"}
    mpiSettings = LaunchSettings(launcher=launcher, env_vars=env_vars)
    formatted = mpiSettings.format_env_vars()
    result = [
        "-x",
        "OMP_NUM_THREADS=20",
        "-x",
        "LOGGING=verbose",
    ]
    assert formatted == result


@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.Mpirun, id="format_launcher_args_mpirun"),
        pytest.param(LauncherType.Mpiexec, id="format_launcher_args_mpiexec"),
        pytest.param(LauncherType.Orterun, id="format_launcher_args_orterun"),
    ],
)
def test_format_launcher_args(launcher):
    mpiSettings = LaunchSettings(launcher=launcher)
    mpiSettings.launch_args.set_cpus_per_task(1)
    mpiSettings.launch_args.set_tasks(2)
    mpiSettings.launch_args.set_hostlist(["node005", "node006"])
    formatted = mpiSettings.format_launch_args()
    result = ["--cpus-per-proc", "1", "--n", "2", "--host", "node005,node006"]
    assert formatted == result


@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.Mpirun, id="set_verbose_launch_mpirun"),
        pytest.param(LauncherType.Mpiexec, id="set_verbose_launch_mpiexec"),
        pytest.param(LauncherType.Orterun, id="set_verbose_launch_orterun"),
    ],
)
def test_set_verbose_launch(launcher):
    mpiSettings = LaunchSettings(launcher=launcher)
    mpiSettings.launch_args.set_verbose_launch(True)
    assert mpiSettings.launch_args._launch_args == {"verbose": None}
    mpiSettings.launch_args.set_verbose_launch(False)
    assert mpiSettings.launch_args._launch_args == {}


@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.Mpirun, id="set_quiet_launch_mpirun"),
        pytest.param(LauncherType.Mpiexec, id="set_quiet_launch_mpiexec"),
        pytest.param(LauncherType.Orterun, id="set_quiet_launch_orterun"),
    ],
)
def test_set_quiet_launch(launcher):
    mpiSettings = LaunchSettings(launcher=launcher)
    mpiSettings.launch_args.set_quiet_launch(True)
    assert mpiSettings.launch_args._launch_args == {"quiet": None}
    mpiSettings.launch_args.set_quiet_launch(False)
    assert mpiSettings.launch_args._launch_args == {}


@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.Mpirun, id="invalid_hostlist_mpirun"),
        pytest.param(LauncherType.Mpiexec, id="invalid_hostlist_mpiexec"),
        pytest.param(LauncherType.Orterun, id="invalid_hostlist_orterun"),
    ],
)
def test_invalid_hostlist_format(launcher):
    """Test invalid hostlist formats"""
    mpiSettings = LaunchSettings(launcher=launcher)
    with pytest.raises(TypeError):
        mpiSettings.launch_args.set_hostlist(["test", 5])
    with pytest.raises(TypeError):
        mpiSettings.launch_args.set_hostlist([5])
    with pytest.raises(TypeError):
        mpiSettings.launch_args.set_hostlist(5)


@pytest.mark.parametrize(
    "cls, cmd",
    (
        pytest.param(MpiArgBuilder, "mpirun", id="w/ mpirun"),
        pytest.param(MpiexecArgBuilder, "mpiexec", id="w/ mpiexec"),
        pytest.param(OrteArgBuilder, "orterun", id="w/ orterun"),
    ),
)
@pytest.mark.parametrize(
    "args, expected",
    (
        pytest.param({}, ("--", "echo", "hello", "world"), id="Empty Args"),
        pytest.param(
            {"n": "1"},
            ("--n", "1", "--", "echo", "hello", "world"),
            id="Short Arg",
        ),
        pytest.param(
            {"host": "myhost"},
            ("--host", "myhost", "--", "echo", "hello", "world"),
            id="Long Arg",
        ),
        pytest.param(
            {"v": None},
            ("--v", "--", "echo", "hello", "world"),
            id="Short Arg (No Value)",
        ),
        pytest.param(
            {"verbose": None},
            ("--verbose", "--", "echo", "hello", "world"),
            id="Long Arg (No Value)",
        ),
        pytest.param(
            {"n": "1", "host": "myhost"},
            ("--n", "1", "--host", "myhost", "--", "echo", "hello", "world"),
            id="Short and Long Args",
        ),
    ),
)
def test_formatting_launch_args(
    echo_executable_like, cls, cmd, args, expected, test_dir
):
    fmt, path = cls(args).finalize(echo_executable_like, {}, test_dir)
    assert tuple(fmt) == (cmd,) + expected
    assert path == test_dir
