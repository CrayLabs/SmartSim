from smartsim.settingshold import LaunchSettings
from smartsim.settingshold.translators.launch.mpi import MpiArgTranslator, MpiexecArgTranslator, OrteArgTranslator
import pytest
import logging
import itertools
    
@pytest.mark.parametrize(
    "l,function,value,result,flag",
    [
    # Use OpenMPI style settigs for all launchers
    *itertools.chain.from_iterable(
        (
            (
            pytest.param(l, "set_task_map", ("taskmap",),"taskmap","map-by",id="set_task_map"),
            pytest.param(l, "set_cpus_per_task", (2,),2,"cpus-per-proc",id="set_cpus_per_task"),
            pytest.param(l, "set_cpu_binding_type", ("4",),"4","bind-to",id="set_cpu_binding_type"),
            pytest.param(l, "set_tasks_per_node", (4,),4,"npernode",id="set_tasks_per_node"),
            pytest.param(l, "set_tasks", (4,),4,"n",id="set_tasks"),
            pytest.param(l, "set_executable_broadcast", ("broadcast",),"broadcast","preload-binary",id="set_executable_broadcast"),
            pytest.param(l, "set_hostlist", ("host_A",),"host_A","host",id="set_hostlist_str"),
            pytest.param(l, "set_hostlist", (["host_A","host_B"],),"host_A,host_B","host",id="set_hostlist_list[str]"),
            pytest.param(l, "set_hostlist_from_file", ("./path/to/hostfile",),"./path/to/hostfile","hostfile",id="set_hostlist_from_file"),
            pytest.param(l, "set_node_feature", ("P100",),id="set_node_feature"),
            pytest.param(l, "set_binding", ("bind",), id="set_binding"),
        )
                for l in ("mpirun", "orterun", "mpiexec")
            ))
    ],
)
def test_update_env_initialized(l,function, value, flag, result):
    mpiSettings = LaunchSettings(launcher=l)
    assert mpiSettings.launcher == l
    getattr(mpiSettings, function)(*value)
    assert mpiSettings.launcher_args[flag] == result

@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param("mpirun", id="format_env"),
        pytest.param("orterun", id="format_env"),
        pytest.param("mpiexec", id="format_env"),
    ],
)
def test_format_env(launcher):
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose"}
    mpiSettings = LaunchSettings(launcher=launcher, env_vars=env_vars)
    assert mpiSettings.launcher == launcher
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
        pytest.param("mpirun", id="format_env"),
        pytest.param("orterun", id="format_env"),
        pytest.param("mpiexec", id="format_env"),
    ],
)
def test_set_verbose_launch(launcher):
    mpiSettings = LaunchSettings(launcher=launcher)
    assert mpiSettings.launcher == launcher
    mpiSettings.set_verbose_launch(True)
    assert mpiSettings.launcher_args == {'verbose': None}
    mpiSettings.set_verbose_launch(False)
    assert mpiSettings.launcher_args == {}

@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param("mpirun", id="format_env"),
        pytest.param("orterun", id="format_env"),
        pytest.param("mpiexec", id="format_env"),
    ],
)
def test_set_quiet_launch(launcher):
    mpiSettings = LaunchSettings(launcher=launcher)
    assert mpiSettings.launcher == launcher
    mpiSettings.set_quiet_launch(True)
    assert mpiSettings.launcher_args == {'quiet': None}
    mpiSettings.set_quiet_launch(False)
    assert mpiSettings.launcher_args == {}

@pytest.mark.parametrize(
    "launcher",
    [
        pytest.param("mpirun", id="format_env"),
        pytest.param("orterun", id="format_env"),
        pytest.param("mpiexec", id="format_env"),
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
        pytest.param("mpirun", id="launcher_str_mpirun"),
        pytest.param("orterun", id="launcher_str_orterun"),
        pytest.param("mpiexec", id="launcher_str_mpiexec"),
    ],
)
def test_launcher_str(launcher):
    mpiLauncher = LaunchSettings(launcher=launcher)
    assert mpiLauncher.launcher_str() == launcher

@pytest.mark.parametrize(
    "l,method,params",
    [
    # Use OpenMPI style settigs for all launchers
    *itertools.chain.from_iterable(
        (   
            (
            pytest.param(l, "set_nodes", (1,), id="set_nodes"),
            pytest.param(l, "set_excluded_hosts", ("hosts",), id="set_excluded_hosts"),
            pytest.param(l, "set_cpu_bindings", (1,), id="set_cpu_bindings"),
            pytest.param(l, "set_memory_per_node", (3000,), id="set_memory_per_node"),
            pytest.param(l, "set_binding", ("bind",), id="set_binding"),
            )
        for l in ("mpirun", "orterun", "mpiexec")
            ))
    ],
)
def test_unimplimented_setters_throw_warning(l, caplog, method, params):
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
                and ("not supported" and l) in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )