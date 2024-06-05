from smartsim.settings import LaunchSettings
from smartsim.settings.translators.launch.lsf import JsrunArgTranslator
from smartsim.settings.launchCommand import LauncherType
import pytest

@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_tasks", (2,),"2","np",id="set_tasks"),
        pytest.param("set_binding", ("packed:21",),"packed:21","bind",id="set_binding"),
    ],
)
def test_lsf_class_methods(function, value, flag, result):
    lsfLauncher = LaunchSettings(launcher=LauncherType.LsfLauncher)
    assert isinstance(lsfLauncher._arg_translator,JsrunArgTranslator)
    getattr(lsfLauncher.launch_args, function)(*value)
    assert lsfLauncher.launch_args._launch_args[flag] == result

def test_format_env_vars():
    env_vars = {"OMP_NUM_THREADS": None, "LOGGING": "verbose"}
    lsfLauncher = LaunchSettings(launcher=LauncherType.LsfLauncher, env_vars=env_vars)
    assert isinstance(lsfLauncher._arg_translator,JsrunArgTranslator)
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
    lsfLauncher = LaunchSettings(launcher=LauncherType.LsfLauncher, launch_args=launch_args)
    assert isinstance(lsfLauncher._arg_translator,JsrunArgTranslator)
    formatted = lsfLauncher.format_launch_args()
    result = [
        "--latency_priority=gpu-gpu",
        "--immediate",
        "-d",
        "packed",
        "--nrs=10",
        "--np=100",
    ]
    assert formatted == result