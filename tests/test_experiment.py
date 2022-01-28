import pytest

from smartsim import Experiment
from smartsim.entity import Model
from smartsim.error import SmartSimError
from smartsim.settings import RunSettings


def test_model_prefix(fileutils):
    exp_name = "test_prefix"
    exp = Experiment(exp_name)
    test_dir = fileutils.make_test_dir(exp_name)
    model = exp.create_model(
        "model",
        path=test_dir,
        run_settings=RunSettings("python"),
        enable_key_prefixing=True,
    )
    assert model._key_prefixing_enabled == True


def test_bad_exp_path():
    with pytest.raises(NotADirectoryError):
        exp = Experiment("test", "not-a-directory")


def test_type_exp_path():
    with pytest.raises(TypeError):
        exp = Experiment("test", ["this-is-a-list-dummy"])


def test_stop_type():
    """Wrong argument type given to stop"""
    exp = Experiment("name")
    with pytest.raises(TypeError):
        exp.stop("model")


def test_finished_new_model():
    # finished should fail as this model hasn't been
    # launched yet.

    model = Model("name", {}, "./", RunSettings("python"))
    exp = Experiment("test")
    with pytest.raises(ValueError):
        exp.finished(model)


def test_status_typeerror():
    exp = Experiment("test")
    with pytest.raises(TypeError):
        exp.get_status([])


def test_status_pre_launch():
    model = Model("name", {}, "./", RunSettings("python"))
    exp = Experiment("test")
    with pytest.raises(SmartSimError):
        exp.get_status(model)


def test_bad_ensemble_init_no_rs():
    """params supplied without run settings"""
    exp = Experiment("test")
    with pytest.raises(SmartSimError):
        exp.create_ensemble("name", {"param1": 1})


def test_bad_ensemble_init_no_params():
    """params supplied without run settings"""
    exp = Experiment("test")
    with pytest.raises(SmartSimError):
        exp.create_ensemble("name", run_settings=RunSettings("python"))


def test_bad_ensemble_init_no_rs_bs():
    """ensemble init without run settings or batch settings"""
    exp = Experiment("test")
    with pytest.raises(SmartSimError):
        exp.create_ensemble("name")


def test_stop_entity(fileutils):
    exp_name = "test_stop_entity"
    exp = Experiment(exp_name)
    test_dir = fileutils.make_test_dir(exp_name)
    m = exp.create_model("model", path=test_dir, run_settings=RunSettings("sleep", "5"))
    exp.start(m, block=False)
    assert exp.finished(m) == False
    exp.stop(m)
    assert exp.finished(m) == True


def test_poll(fileutils):
    # Ensure that a SmartSimError is not raised
    exp_name = "test_exp_poll"
    exp = Experiment(exp_name)
    test_dir = fileutils.make_test_dir(exp_name)
    model = exp.create_model(
        "model", path=test_dir, run_settings=RunSettings("sleep", "5")
    )
    exp.start(model, block=False)
    exp.poll(interval=1)
    exp.stop(model)


def test_summary(fileutils):
    exp_name = "test_exp_summary"
    exp = Experiment(exp_name)
    test_dir = fileutils.make_test_dir(exp_name)
    m = exp.create_model(
        "model", path=test_dir, run_settings=RunSettings("echo", "Hello")
    )
    exp.start(m)
    summary_str = exp.summary(format="plain")
    print(summary_str)

    summary_lines = summary_str.split("\n")
    assert 2 == len(summary_lines)

    headers, values = [s.split() for s in summary_lines]
    headers = ["Index"] + headers

    row = dict(zip(headers, values))
    assert m.name == row["Name"]
    assert m.type == row["Entity-Type"]
    assert 0 == int(row["RunID"])
    assert 0 == int(row["Returncode"])


def test_launcher_detection(wlmutils):
    exp = Experiment("test-launcher-detection", launcher="auto")

    # We check whether the right launcher is found. But if
    # the test launcher was set to local, we tolerate finding
    # another one (this cannot be avoided)
    if (
        exp._launcher != wlmutils.get_test_launcher()
        and wlmutils.get_test_launcher() != "local"
    ):
        assert False
