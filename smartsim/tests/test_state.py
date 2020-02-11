
from smartsim import Generator, State
from os import path, environ, getcwd
from shutil import rmtree
from ..error import SmartSimError, SSModelExistsError
from ..model import NumModel
from ..ensemble import Ensemble
import pytest

def test_create_model():
    state = State(experiment="test")
    state.create_model("test-model")
    assert(type(state.ensembles[0].models["test-model"]) == NumModel)

def test_create_model_in_ensemble():
    state = State(experiment="test")
    state.create_ensemble("test-ensemble")
    state.create_model("test-model", "test-ensemble")
    assert(type(state.ensembles[0].models["test-model"]) == NumModel)

def test_get_model():
    state = State(experiment="test")
    state.create_ensemble("test-ensemble")
    state.create_model("test-model", "test-ensemble")
    assert(state.ensembles[0].models["test-model"] == state.get_model("test-model", "test-ensemble"))

def test_create_ensemble():
    state = State(experiment="test")
    state.create_ensemble("test-ensemble")
    assert(len(state.ensembles) > 0)
    assert(type(state.ensembles[0]) == Ensemble)

def test_get_ensemble():
    state = State(experiment="test")
    ensemble = state.create_ensemble("test-ensemble")
    assert(ensemble == state.ensembles[0])

def test_get_ensemble_error():
    state = State(experiment="test")
    state.create_ensemble("test-ensemble")
    try:
        ensemble = state.get_ensemble("test-ensemble_doesnt_exist")
        assert(False)
    except SmartSimError:
        assert(True)

def test_delete_ensemble():
    state = State(experiment="test")
    state.create_ensemble("test-ensemble")
    assert(len(state.ensembles) > 0)
    assert(type(state.ensembles[0]) == Ensemble)
    state.delete_ensemble("test-ensemble")
    assert(len(state.ensembles) == 0)

def test_create_node():
    state = State(experiment="test")
    state.create_node("test-node", script_path=getcwd())
    assert(len(state.nodes) > 0)

def test_ensemble_get_model():
    state = State(experiment="test")
    ensemble = state.create_ensemble("test-ensemble")
    model = state.create_model("test-model", "test-ensemble")
    assert(model == ensemble["test-model"])

def test_duplicate_orchestrator_error():
    state = State(experiment="test")
    state.create_orchestrator(name='first_orc')

    with pytest.raises(SmartSimError):
        state.create_orchestrator(name="double_orc")
