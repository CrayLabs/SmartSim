import pytest

from smartsim import Experiment
from smartsim.entity import Model
from smartsim.settings import RunSettings
from smartsim.error import SmartSimError


def test_model_prefix():
    exp = Experiment("test")
    model = exp.create_model("model", RunSettings("python"), enable_key_prefixing=True)
    assert(model._key_prefixing_enabled == True)

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

def test_finished_type():
    model = Model("name", {}, "./", RunSettings("python"))
    exp = Experiment("test")
    with pytest.raises(SmartSimError):
        exp.finished(model)

def test_status_type():
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




