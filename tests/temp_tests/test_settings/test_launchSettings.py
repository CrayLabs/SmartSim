from smartsim.settings import LaunchSettings
from smartsim.settings.launchCommand import LauncherType
import pytest
import logging

@pytest.mark.parametrize(
    "launch_enum",
    [
        pytest.param(LauncherType.SlurmLauncher,id="slurm"),
        pytest.param(LauncherType.DragonLauncher,id="dragon"),
        pytest.param(LauncherType.PalsLauncher,id="pals"),
        pytest.param(LauncherType.AlpsLauncher,id="alps"),
        pytest.param(LauncherType.LocalLauncher,id="local"),
        pytest.param(LauncherType.MpiexecLauncher,id="mpiexec"),
        pytest.param(LauncherType.MpirunLauncher,id="mpirun"),
        pytest.param(LauncherType.OrterunLauncher,id="orterun"),
        pytest.param(LauncherType.LsfLauncher,id="lsf"),
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

# TODO need to test launch_args
# def test_set_launch_args():
#     ls = LaunchSettings(launcher="local", launch_args = {"init":"arg"})
#     assert ls.launch_args == {"init":"arg"}
#     ls.launch_args = {"launch":"arg"}
#     assert ls.launch_args == {"launch":"arg"}

# @pytest.mark.parametrize(
#     "launcher,key",
#     [
#         pytest.param(LauncherType.SlurmLauncher, ("chdir",), id="slurm-chdir"),
#         pytest.param(LauncherType.SlurmLauncher, ("D",), id="slurm-D"),
#         pytest.param(LauncherType.LsfLauncher, ("chdir",), id="lsf-chdir"),
#         pytest.param(LauncherType.LsfLauncher, ("h",), id="lsf-h"),
#         pytest.param(LauncherType.MpiexecLauncher, ("wd",), id="mpiexec-wd"),
#         pytest.param(LauncherType.OrterunLauncher, ("wd",), id="orte-wd"),
#         pytest.param(LauncherType.MpirunLauncher, ("wd",), id="mpi-wd"),
#         pytest.param(LauncherType.MpiexecLauncher, ("wdir",), id="mpiexec-wdir"),
#         pytest.param(LauncherType.OrterunLauncher, ("wdir",), id="orte-wdir"),
#         pytest.param(LauncherType.MpirunLauncher, ("wdir",), id="mpi-wdir"),
#     ],
# )
# def test_prevent_set_reserved_launch_args(caplog, launcher, key):
#     """Test methods not implemented throw warnings"""
#     from smartsim.settings.launchSettings import logger

#     prev_prop = logger.propagate
#     logger.propagate = True

#     with caplog.at_level(logging.WARNING):
#         caplog.clear()
#         launchSettings = LaunchSettings(launcher=launcher)
#         try:
#             getattr(launchSettings, "set")(*key, None)
#         finally:
#             logger.propagate = prev_prop

#         for rec in caplog.records:
#             if (
#                 logging.WARNING <= rec.levelno < logging.ERROR
#                 and "Could not set argument" in rec.msg
#             ):
#                 break
#         else:
#             pytest.fail(
#                 (
#                     f"No message stating method `{key}` is not "
#                     "implemented at `warning` level"
#                 )
#             )

# def test_log_overwrite_set_warning_message(caplog):
#     """Test methods not implemented throw warnings"""
#     from smartsim.settings.launchSettings import logger

#     prev_prop = logger.propagate
#     logger.propagate = True

#     with caplog.at_level(logging.WARNING):
#         caplog.clear()
#         launchSettings = LaunchSettings(launcher=LauncherType.LocalLauncher)
#         launchSettings.set("test", None)
#         try:
#             getattr(launchSettings, "set")("test", "overwritting")
#         finally:
#             logger.propagate = prev_prop

#         for rec in caplog.records:
#             if (
#                 logging.WARNING <= rec.levelno < logging.ERROR
#                 and "Overwritting argument" in rec.msg
#             ):
#                 break
#         else:
#             pytest.fail(
#                 (
#                     f"No message stating method `test` will be "
#                     "overwritten at `warning` level"
#                 )
#             )