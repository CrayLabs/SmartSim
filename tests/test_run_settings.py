from pprint import pformat
from shutil import which

import pytest

from smartsim.settings import MpirunSettings, RunSettings
from smartsim.settings.settings import create_run_settings


def test_create_run_settings_local():
    # no run command provided
    rs = create_run_settings("local", "echo", "hello", run_command=None)
    assert rs.run_command == None
    assert type(rs) == RunSettings

    # auto should never return a run_command when
    # the user has specified the local launcher
    auto = create_run_settings("local", "echo", "hello", run_command="auto")
    assert auto.run_command == None
    assert type(auto) == RunSettings

    # Test when a run_command is provided that we do not currently have a helper
    # implementation for it.
    # NOTE: we allow for the command to be invalid if it's user specified in the
    # case where a head node may not have the same installed binaries as the MOM
    # or compute nodes.
    specific = create_run_settings("local", "echo", "hello", run_command="specific")
    assert specific.run_command == "specific"
    assert type(specific) == RunSettings

    # make it return MpirunSettings
    _mpirun = which("mpirun")
    if _mpirun:
        mpirun = create_run_settings("local", "echo", "hello", run_command="mpirun")
        assert mpirun.run_command == _mpirun
        assert type(mpirun) == MpirunSettings


####### Base Run Settings tests #######


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

def test_set_args():
    rs = RunSettings("python")
    rs.set("str", "some-string")
    rs.set("nothing")
    
    assert "str" in rs.run_args
    assert rs.run_args["str"] == "some-string"
    
    assert "nothing" in rs.run_args
    assert rs.run_args["nothing"] is None

@pytest.mark.parametrize("set_str,val,key", [
    pytest.param("normal-key", "some-val", "normal-key", id="set string"),
    pytest.param("--a-key", "a-value", "a-key", id="strip doulbe dashes"),
    pytest.param("-b", "some-str", "b", id="strip single dashes"),
    pytest.param("   c    ", "some-val", "c", id="strip spaces"),
    pytest.param("   --a-mess    ", "5", "a-mess", id="strip everything"),
])
def test_set_format_args(set_str, val, key):
    rs = RunSettings("python")
    rs.set(set_str, val)
    assert rs.run_args[key] == val

def test_set_raises_type_errors():
    rs = RunSettings("python")
    
    with pytest.raises(TypeError):
        rs.set("good-key", 5)
    
    with pytest.raises(TypeError):
        rs.set(9)

def test_set_overwrites_prev_args():
    rs = RunSettings("python")
    rs.set("some-key", "some-val")
    rs.set("some-key", "another-val")
    assert rs.run_args["some-key"] == "another-val"
