from smartsim.settings import LaunchSettings
from smartsim.settings.launchCommand import LauncherType
import pytest
import logging

@pytest.mark.parametrize(
    "launch_enum",
    [
        pytest.param(LauncherType.Slurm,id="slurm"),
        pytest.param(LauncherType.Dragon,id="dragon"),
        pytest.param(LauncherType.Pals,id="pals"),
        pytest.param(LauncherType.Alps,id="alps"),
        pytest.param(LauncherType.Local,id="local"),
        pytest.param(LauncherType.Mpiexec,id="mpiexec"),
        pytest.param(LauncherType.Mpirun,id="mpirun"),
        pytest.param(LauncherType.Orterun,id="orterun"),
        pytest.param(LauncherType.Lsf,id="lsf"),
    ],
)
def test_create_launch_settings(launch_enum):
    ls_str = LaunchSettings(launcher=launch_enum.value, launch_args={"launch":"var"}, env_vars={"ENV":"VAR"})
    assert ls_str._launcher == launch_enum
    # TODO need to test launch_args
    assert ls_str._env_vars == {"ENV":"VAR"}
    
    ls_enum = LaunchSettings(launcher=launch_enum, launch_args={"launch":"var"}, env_vars={"ENV":"VAR"})
    assert ls_enum._launcher == launch_enum
    # TODO need to test launch_args
    assert ls_enum._env_vars == {"ENV":"VAR"}

def test_launcher_property():
    ls = LaunchSettings(launcher="local")
    assert ls.launcher == "local"

def test_env_vars_property():
    ls = LaunchSettings(launcher="local", env_vars={"ENV":"VAR"})
    assert ls.env_vars == {"ENV":"VAR"}

def test_env_vars_property_deep_copy():
    ls = LaunchSettings(launcher="local", env_vars={"ENV":"VAR"})
    copy_env_vars = ls.env_vars
    copy_env_vars.update({"test":"no_update"})
    assert ls.env_vars == {"ENV":"VAR"}

def test_update_env_vars():
    ls = LaunchSettings(launcher="local", env_vars={"ENV":"VAR"})
    ls.update_env({"test":"no_update"})
    assert ls.env_vars == {"ENV":"VAR","test":"no_update"}
    
def test_update_env_vars_errors():
    ls = LaunchSettings(launcher="local", env_vars={"ENV":"VAR"})
    with pytest.raises(TypeError):
        ls.update_env({"test":1})
    with pytest.raises(TypeError):
        ls.update_env({1:"test"})
    with pytest.raises(TypeError):
        ls.update_env({1:1})
    with pytest.raises(TypeError):
        # Make sure the first key and value do not assign
        # and that the function is atomic
        ls.update_env({"test":"test","test":1})
        assert ls.env_vars == {"ENV":"VAR"}