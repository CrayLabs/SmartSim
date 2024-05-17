from smartsim.settingshold import LaunchSettings
from smartsim.settingshold.translators.launch.slurm import SlurmArgTranslator
from smartsim.settingshold.launchCommand import LauncherType
import pytest
import logging

def test_launcher_str():
    """Ensure launcher_str returns appropriate value"""
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher)
    assert slurmLauncher.launcher_str() == LauncherType.SlurmLauncher.value

@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,),2,"nodes",id="set_nodes"),
        pytest.param("set_hostlist", ("host_A",),"host_A","nodelist",id="set_hostlist_str"),
        pytest.param("set_hostlist", (["host_A","host_B"],),"host_A,host_B","nodelist",id="set_hostlist_list[str]"),
        pytest.param("set_hostlist_from_file", ("./path/to/hostfile",),"./path/to/hostfile","nodefile",id="set_hostlist_from_file"),
        pytest.param("set_excluded_hosts", ("host_A",),"host_A","exclude",id="set_excluded_hosts_str"),
        pytest.param("set_excluded_hosts", (["host_A","host_B"],),"host_A,host_B","exclude",id="set_excluded_hosts_list[str]"),
        pytest.param("set_cpus_per_task", (4,),4,"cpus-per-task",id="set_cpus_per_task"),
        pytest.param("set_tasks", (4,),4,"ntasks",id="set_tasks"),
        pytest.param("set_tasks_per_node", (4,),4,"ntasks-per-node",id="set_tasks_per_node"),
        pytest.param("set_cpu_bindings", (4,),"map_cpu:4","cpu_bind",id="set_cpu_bindings"),
        pytest.param("set_cpu_bindings", ([4,4],),"map_cpu:4,4","cpu_bind",id="set_cpu_bindings_list[str]"),
        pytest.param("set_memory_per_node", (8000,),"8000M","mem",id="set_memory_per_node"),
        pytest.param("set_executable_broadcast", ("/tmp/some/path",),"/tmp/some/path","bcast",id="set_broadcast"),
        pytest.param("set_node_feature", ("P100",),"P100","C",id="set_node_feature"),
        pytest.param("set_walltime", ("10:00:00",),"10:00:00","time",id="set_walltime"),
        pytest.param("set_verbose_launch", (True,),None,"verbose",id="set_walltime"),
    ],
)
def test_update_env_initialized(function, value, flag, result):
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher)
    assert slurmLauncher.launcher.value == LauncherType.SlurmLauncher.value
    assert isinstance(slurmLauncher.arg_translator,SlurmArgTranslator)
    getattr(slurmLauncher, function)(*value)
    assert slurmLauncher.launcher_args[flag] == result

def test_set_verbose_launch():
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher)
    assert slurmLauncher.launcher.value == LauncherType.SlurmLauncher.value
    assert isinstance(slurmLauncher.arg_translator,SlurmArgTranslator)
    slurmLauncher.set_verbose_launch(True)
    assert slurmLauncher.launcher_args == {'verbose': None}
    slurmLauncher.set_verbose_launch(False)
    assert slurmLauncher.launcher_args == {}

def test_set_quiet_launch():
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher)
    assert slurmLauncher.launcher.value == LauncherType.SlurmLauncher.value
    assert isinstance(slurmLauncher.arg_translator,SlurmArgTranslator)
    slurmLauncher.set_quiet_launch(True)
    assert slurmLauncher.launcher_args == {'quiet': None}
    slurmLauncher.set_quiet_launch(False)
    assert slurmLauncher.launcher_args == {}
    
def test_format_env_vars():
    """Test format_env_vars runs correctly"""
    env_vars={
        "OMP_NUM_THREADS": 20,
        "LOGGING": "verbose",
        "SSKEYIN": "name_0,name_1",
    }
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher, env_vars=env_vars)
    assert slurmLauncher.launcher.value == LauncherType.SlurmLauncher.value
    assert isinstance(slurmLauncher.arg_translator,SlurmArgTranslator)
    formatted = slurmLauncher.format_env_vars()
    assert "OMP_NUM_THREADS=20" in formatted
    assert "LOGGING=verbose" in formatted
    assert all("SSKEYIN" not in x for x in formatted)

def test_format_comma_sep_env_vars():
    """Test format_comma_sep_env_vars runs correctly"""
    env_vars = {"OMP_NUM_THREADS": 20, "LOGGING": "verbose", "SSKEYIN": "name_0,name_1"}
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher, env_vars=env_vars)
    formatted, comma_separated_formatted = slurmLauncher.format_comma_sep_env_vars()
    assert "OMP_NUM_THREADS" in formatted
    assert "LOGGING" in formatted
    assert "SSKEYIN" in formatted
    assert "name_0,name_1" not in formatted
    assert "SSKEYIN=name_0,name_1" in comma_separated_formatted

def test_srun_settings():
    """Test format_launcher_args runs correctly"""
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher)
    slurmLauncher.set_nodes(5)
    slurmLauncher.set_cpus_per_task(2)
    slurmLauncher.set_tasks(100)
    slurmLauncher.set_tasks_per_node(20)
    formatted = slurmLauncher.format_launcher_args()
    result = ["--nodes=5", "--cpus-per-task=2", "--ntasks=100", "--ntasks-per-node=20"]
    assert formatted == result

def test_srun_launcher_args():
    """Test the possible user overrides through run_args"""
    launcher_args = {
        "account": "A3123",
        "exclusive": None,
        "C": "P100",  # test single letter variables
        "nodes": 10,
        "ntasks": 100,
    }
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher, launcher_args=launcher_args)
    formatted = slurmLauncher.format_launcher_args()
    result = [
        "--account=A3123",
        "--exclusive",
        "-C",
        "P100",
        "--nodes=10",
        "--ntasks=100",
    ]
    assert formatted == result

def test_invalid_hostlist_format():
    """Test invalid hostlist formats"""
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher)
    with pytest.raises(TypeError):
        slurmLauncher.set_hostlist(["test",5])
    with pytest.raises(TypeError):
        slurmLauncher.set_hostlist([5])
    with pytest.raises(TypeError):
        slurmLauncher.set_hostlist(5)

def test_invalid_exclude_hostlist_format():
    """Test invalid hostlist formats"""
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher)
    with pytest.raises(TypeError):
        slurmLauncher.set_excluded_hosts(["test",5])
    with pytest.raises(TypeError):
        slurmLauncher.set_excluded_hosts([5])
    with pytest.raises(TypeError):
        slurmLauncher.set_excluded_hosts(5)

def test_invalid_node_feature_format():
    """Test invalid node feature formats"""
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher)
    with pytest.raises(TypeError):
        slurmLauncher.set_node_feature(["test",5])
    with pytest.raises(TypeError):
        slurmLauncher.set_node_feature([5])
    with pytest.raises(TypeError):
        slurmLauncher.set_node_feature(5)

def test_invalid_walltime_format():
    """Test invalid walltime formats"""
    slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher)
    with pytest.raises(ValueError):
        slurmLauncher.set_walltime("11:11")
    with pytest.raises(ValueError):
        slurmLauncher.set_walltime("ss:ss:ss")
    with pytest.raises(ValueError):
        slurmLauncher.set_walltime("11:ss:ss")
    with pytest.raises(ValueError):
        slurmLauncher.set_walltime("0s:ss:ss")

@pytest.mark.parametrize(
    "method,params",
    [
        pytest.param("set_cpu_binding_type", ("bind",), id="set_cpu_binding_type"),
        pytest.param("set_task_map", ("task:map",), id="set_task_map"),
        pytest.param("set_binding", ("bind",), id="set_binding"),
    ],
)
def test_unimplimented_methods_throw_warning(caplog, method, params):
    """Test methods not implemented throw warnings"""
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        slurmLauncher = LaunchSettings(launcher=LauncherType.SlurmLauncher)
        try:
            getattr(slurmLauncher, method)(*params)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and (method and "not supported" and "slurm") in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{method}` is not "
                    "implemented at `warning` level"
                )
            )