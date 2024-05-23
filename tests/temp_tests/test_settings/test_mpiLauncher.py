from smartsim.settingshold import LaunchSettings
from smartsim.settingshold.translators.launch.mpi import MpiArgTranslator, MpiexecArgTranslator, OrteArgTranslator
import pytest
import logging
import itertools
from smartsim.settingshold.launchCommand import LauncherType

@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.MpirunLauncher, id="format_launcher_args_mpirun"),
        pytest.param(LauncherType.MpiexecLauncher, id="format_launcher_args_mpiexec"),
        pytest.param(LauncherType.OrterunLauncher, id="format_launcher_args_orterun"),
    ],
)
def test_launcher_str(launcher):
    """Ensure launcher_str returns appropriate value"""
    mpiSettings = LaunchSettings(launcher=launcher)
    assert mpiSettings.launcher_str() == LauncherType.PalsLauncher.value

@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.MpirunLauncher, id="format_launcher_args_mpirun"),
        pytest.param(LauncherType.MpiexecLauncher, id="format_launcher_args_mpiexec"),
        pytest.param(LauncherType.OrterunLauncher, id="format_launcher_args_orterun"),
    ],
)
def test_set_reserved_launcher_args(launcher):
    """Ensure launcher_str returns appropriate value"""
    mpiSettings = LaunchSettings(launcher=launcher)
    assert mpiSettings._reserved_launch_args == {"wd", "wdir"}

@pytest.mark.parametrize(
    "l,function,value,result,flag",
    [
    # Use OpenMPI style settigs for all launchers
    *itertools.chain.from_iterable(
        (
            (
            pytest.param(l, "set_walltime", ("100",),"100","timeout",id="set_walltime"),
            pytest.param(l, "set_task_map", ("taskmap",),"taskmap","map-by",id="set_task_map"),
            pytest.param(l, "set_cpus_per_task", (2,),2,"cpus-per-proc",id="set_cpus_per_task"),
            pytest.param(l, "set_cpu_binding_type", ("4",),"4","bind-to",id="set_cpu_binding_type"),
            pytest.param(l, "set_tasks_per_node", (4,),4,"npernode",id="set_tasks_per_node"),
            pytest.param(l, "set_tasks", (4,),4,"n",id="set_tasks"),
            pytest.param(l, "set_executable_broadcast", ("broadcast",),"broadcast","preload-binary",id="set_executable_broadcast"),
            pytest.param(l, "set_hostlist", ("host_A",),"host_A","host",id="set_hostlist_str"),
            pytest.param(l, "set_hostlist", (["host_A","host_B"],),"host_A,host_B","host",id="set_hostlist_list[str]"),
            pytest.param(l, "set_hostlist_from_file", ("./path/to/hostfile",),"./path/to/hostfile","hostfile",id="set_hostlist_from_file"),
        )
                for l in ([LauncherType.MpirunLauncher, MpiArgTranslator], [LauncherType.MpiexecLauncher, MpiexecArgTranslator], [LauncherType.OrterunLauncher, OrteArgTranslator])
            ))
    ],
)
def test_mpi_class_methods(l,function, value, flag, result):
    mpiSettings = LaunchSettings(launcher=l[0])
    assert isinstance(mpiSettings.arg_translator,l[1])
    assert mpiSettings.launcher.value == l[0].value
    getattr(mpiSettings, function)(*value)
    assert mpiSettings.launcher_args[flag] == result

@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.MpirunLauncher, id="format_env_mpirun"),
        pytest.param(LauncherType.MpiexecLauncher, id="format_env_mpiexec"),
        pytest.param(LauncherType.OrterunLauncher, id="format_env_orterun"),
    ],
)
def test_format_env_vars(launcher):
    env_vars = {"OMP_NUM_THREADS": "20", "LOGGING": "verbose"}
    mpiSettings = LaunchSettings(launcher=launcher, env_vars=env_vars)
    assert mpiSettings.launcher.value == launcher.value
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
        pytest.param(LauncherType.MpirunLauncher, id="format_launcher_args_mpirun"),
        pytest.param(LauncherType.MpiexecLauncher, id="format_launcher_args_mpiexec"),
        pytest.param(LauncherType.OrterunLauncher, id="format_launcher_args_orterun"),
    ],
)
def test_format_launcher_args(launcher):
    mpiSettings = LaunchSettings(launcher=launcher)
    mpiSettings.set_cpus_per_task(1)
    mpiSettings.set_tasks(2)
    mpiSettings.set_hostlist(["node005", "node006"])
    formatted = mpiSettings.format_launcher_args()
    result = ["--cpus-per-proc", "1", "--n", "2", "--host", "node005,node006"]
    assert formatted == result

@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.MpirunLauncher, id="set_verbose_launch_mpirun"),
        pytest.param(LauncherType.MpiexecLauncher, id="set_verbose_launch_mpiexec"),
        pytest.param(LauncherType.OrterunLauncher, id="set_verbose_launch_orterun"),
    ],
)
def test_set_verbose_launch(launcher):
    mpiSettings = LaunchSettings(launcher=launcher)
    assert mpiSettings.launcher.value == launcher.value
    mpiSettings.set_verbose_launch(True)
    assert mpiSettings.launcher_args == {'verbose': None}
    mpiSettings.set_verbose_launch(False)
    assert mpiSettings.launcher_args == {}

@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.MpirunLauncher, id="set_quiet_launch_mpirun"),
        pytest.param(LauncherType.MpiexecLauncher, id="set_quiet_launch_mpiexec"),
        pytest.param(LauncherType.OrterunLauncher, id="set_quiet_launch_orterun"),
    ],
)
def test_set_quiet_launch(launcher):
    mpiSettings = LaunchSettings(launcher=launcher)
    assert mpiSettings.launcher.value == launcher.value
    mpiSettings.set_quiet_launch(True)
    assert mpiSettings.launcher_args == {'quiet': None}
    mpiSettings.set_quiet_launch(False)
    assert mpiSettings.launcher_args == {}

@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.MpirunLauncher, id="invalid_hostlist_mpirun"),
        pytest.param(LauncherType.MpiexecLauncher, id="invalid_hostlist_mpiexec"),
        pytest.param(LauncherType.OrterunLauncher, id="invalid_hostlist_orterun"),
    ],
)
def test_invalid_hostlist_format(launcher):
    """Test invalid hostlist formats"""
    mpiSettings = LaunchSettings(launcher=launcher)
    with pytest.raises(TypeError):
        mpiSettings.set_hostlist(["test",5])
    with pytest.raises(TypeError):
        mpiSettings.set_hostlist([5])
    with pytest.raises(TypeError):
        mpiSettings.set_hostlist(5)

@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param(LauncherType.MpirunLauncher, id="launcher_str_mpirun"),
        pytest.param(LauncherType.MpiexecLauncher, id="launcher_str_mpiexec"),
        pytest.param(LauncherType.OrterunLauncher, id="launcher_str_orterun"),
    ],
)
def test_launcher_str(launcher):
    mpiLauncher = LaunchSettings(launcher=launcher)
    assert mpiLauncher.launcher_str() == launcher.value

@pytest.mark.parametrize(
    "l,method,params",
    [
    *itertools.chain.from_iterable(
        (   
            (
            pytest.param(l, "set_nodes", (1,), id="set_nodes"),
            pytest.param(l, "set_excluded_hosts", ("hosts",), id="set_excluded_hosts"),
            pytest.param(l, "set_cpu_bindings", (1,), id="set_cpu_bindings"),
            pytest.param(l, "set_memory_per_node", (3000,), id="set_memory_per_node"),
            pytest.param(l, "set_binding", ("bind",), id="set_binding"),
            pytest.param(l, "set_node_feature", ("P100",), id="set_node_feature"),
            pytest.param(l, "format_comma_sep_env_vars", (), id="format_comma_sep_env_vars"),
            pytest.param(l, "set_het_group", ([1,2,3,4],), id="set_het_group"),
            )
            for l in (LauncherType.MpirunLauncher, LauncherType.MpiexecLauncher, LauncherType.OrterunLauncher)
            ))
    ],
)
def test_unimplimented_methods_throw_warning(l, caplog, method, params):
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        mpiSettings = LaunchSettings(launcher=l)
        try:
            getattr(mpiSettings, method)(*params)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and (method and "not supported" and l.value) in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )