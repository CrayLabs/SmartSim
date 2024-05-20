from smartsim.settingshold import LaunchSettings
from smartsim.settingshold.launchCommand import LauncherType
import pytest
import logging

def test_set_launch_args():
    launchSettings = LaunchSettings(launcher=LauncherType.LocalLauncher)
    launchSettings.set("str", "some-string")
    launchSettings.set("nothing", None)

    assert "str" in launchSettings.launcher_args
    assert launchSettings.launcher_args["str"] == "some-string"

    assert "nothing" in launchSettings.launcher_args
    assert launchSettings.launcher_args["nothing"] is None

@pytest.mark.parametrize(
    "set_str,val,key",
    [
        pytest.param("normal-key", "some-val", "normal-key", id="set string"),
        pytest.param("--a-key", "a-value", "a-key", id="strip doulbe dashes"),
        pytest.param("-b", "some-str", "b", id="strip single dashes"),
        pytest.param("   c    ", "some-val", "c", id="strip spaces"),
        pytest.param("   --a-mess    ", "5", "a-mess", id="strip everything"),
    ],
)
def test_set_format_args(set_str, val, key):
    launchSettings = LaunchSettings(launcher=LauncherType.LocalLauncher)
    launchSettings.set(set_str, val)
    assert launchSettings.launcher_args[key] == val

def test_set_raises_key_error():
    launchSettings = LaunchSettings(launcher=LauncherType.LocalLauncher)
    with pytest.raises(TypeError):
        launchSettings.set(1, "test")

def test_incorrect_env_var_type():
    with pytest.raises(TypeError):
        _ = LaunchSettings(launcher=LauncherType.LocalLauncher, env_vars={"str": 2})
    with pytest.raises(TypeError):
        _ = LaunchSettings(launcher=LauncherType.LocalLauncher, env_vars={"str": 2.0})
    with pytest.raises(TypeError):
        _ = LaunchSettings(launcher=LauncherType.LocalLauncher, env_vars={"str": "str", "str": 2.0})

def test_incorrect_launch_arg_type():
    with pytest.raises(TypeError):
        _ = LaunchSettings(launcher=LauncherType.LocalLauncher, launcher_args={"str": [1,2]})
    with pytest.raises(TypeError):
        _ = LaunchSettings(launcher=LauncherType.LocalLauncher, launcher_args={"str": LauncherType.LocalLauncher})

@pytest.mark.parametrize(
    "launcher,key",
    [
        pytest.param(LauncherType.SlurmLauncher, ("chdir",), id="slurm-chdir"),
        pytest.param(LauncherType.SlurmLauncher, ("D",), id="slurm-D"),
        pytest.param(LauncherType.LsfLauncher, ("chdir",), id="lsf-chdir"),
        pytest.param(LauncherType.LsfLauncher, ("h",), id="lsf-h"),
        pytest.param(LauncherType.MpiexecLauncher, ("wd",), id="mpiexec-wd"),
        pytest.param(LauncherType.OrterunLauncher, ("wd",), id="orte-wd"),
        pytest.param(LauncherType.MpirunLauncher, ("wd",), id="mpi-wd"),
        pytest.param(LauncherType.MpiexecLauncher, ("wdir",), id="mpiexec-wdir"),
        pytest.param(LauncherType.OrterunLauncher, ("wdir",), id="orte-wdir"),
        pytest.param(LauncherType.MpirunLauncher, ("wdir",), id="mpi-wdir"),
    ],
)
def test_prevent_set_reserved_launch_args(caplog, launcher, key):
    """Test methods not implemented throw warnings"""
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        launchSettings = LaunchSettings(launcher=launcher)
        try:
            getattr(launchSettings, "set")(*key, None)
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and "Could not set argument" in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `{key}` is not "
                    "implemented at `warning` level"
                )
            )

def test_log_overwrite_set_warning_message(caplog):
    """Test methods not implemented throw warnings"""
    from smartsim.settings.base import logger

    prev_prop = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        launchSettings = LaunchSettings(launcher=LauncherType.LocalLauncher)
        launchSettings.set("test", None)
        try:
            getattr(launchSettings, "set")("test", "overwritting")
        finally:
            logger.propagate = prev_prop

        for rec in caplog.records:
            if (
                logging.WARNING <= rec.levelno < logging.ERROR
                and "Overwritting argument" in rec.msg
            ):
                break
        else:
            pytest.fail(
                (
                    f"No message stating method `test` will be "
                    "overwritten at `warning` level"
                )
            )