from shutil import which

import pytest

from smartsim.settings import RunSettings


def test_run_command_exists():
    # you wouldn't actually do this, but we know python will be installed
    settings = RunSettings("python", run_command="python")
    python = which("python")
    assert settings.run_command == python


def test_run_command_not_exists():
    settings = RunSettings("python", run_command="not-on-system")
    assert settings.run_command == "not-on-system"


def test_add_exe_args():
    settings = RunSettings("python")
    settings.add_exe_args("--time 5")
    settings.add_exe_args(["--add", "--list"])
    result = ["--time", "5", "--add", "--list"]
    assert settings.exe_args == result
    with pytest.raises(TypeError):
        settings.add_exe_args([1, 2, 3])


def test_format_run_args():
    settings = RunSettings(
        "echo", exe_args="test", run_command="mpirun", run_args={"-np": 2}
    )
    run_args = settings.format_run_args()
    assert type(run_args) == type(list())
    assert run_args == ["-np", "2"]


def test_addto_existing_exe_args():
    list_exe_args_settings = RunSettings("python", ["sleep.py", "--time=5"])
    str_exe_args_settings = RunSettings("python", "sleep.py --time=5")

    # both should be the same
    args = ["sleep.py", "--time=5"]
    assert list_exe_args_settings.exe_args == args
    assert str_exe_args_settings.exe_args == args

    # add to exe_args
    list_exe_args_settings.add_exe_args("--stop=10")
    str_exe_args_settings.add_exe_args(["--stop=10"])

    args = ["sleep.py", "--time=5", "--stop=10"]
    assert list_exe_args_settings.exe_args == args
    assert str_exe_args_settings.exe_args == args


def test_bad_exe_args():
    """test when user provides incorrect types to exe_args"""
    exe_args = {"dict": "is-wrong-type"}
    with pytest.raises(TypeError):
        _ = RunSettings("python", exe_args=exe_args)


def test_bad_exe_args_2():
    """test when user provides incorrect types to exe_args"""
    exe_args = ["list-includes-int", 5]
    with pytest.raises(TypeError):
        _ = RunSettings("python", exe_args=exe_args)
