
from smartsim import Generator, State
from os import path, environ, getcwd
from shutil import rmtree
from ..error import SmartSimError, SSModelExistsError
from ..model import NumModel
from ..target import Target


def test_create_model():
    state = State(experiment="test")
    state.create_model("test-model")
    assert(type(state.targets[0].models["test-model"]) == NumModel)

def test_create_model_in_target():
    state = State(experiment="test")
    state.create_target("test-target")
    state.create_model("test-model", "test-target")
    assert(type(state.targets[0].models["test-model"]) == NumModel)

def test_get_model():
    state = State(experiment="test")
    state.create_target("test-target")
    state.create_model("test-model", "test-target")
    assert(state.targets[0].models["test-model"] == state.get_model("test-model", "test-target"))

def test_create_target():
    state = State(experiment="test")
    state.create_target("test-target")
    assert(len(state.targets) > 0)
    assert(type(state.targets[0]) == Target)

def test_get_target():
    state = State(experiment="test")
    state.create_target("test-target")
    target = state.get_target("test-target")
    assert(target == state.targets[0])

def test_get_target_error():
    state = State(experiment="test")
    state.create_target("test-target")
    try:
        target = state.get_target("test-target_doesnt_exist")
        assert(False)
    except SmartSimError:
        assert(True)

def test_delete_target():
    state = State(experiment="test")
    state.create_target("test-target")
    assert(len(state.targets) > 0)
    assert(type(state.targets[0]) == Target)
    state.delete_target("test-target")
    assert(len(state.targets) == 0)

def test_create_node():
    state = State(experiment="test")
    state.create_node("test-node", getcwd())
    assert(len(state.nodes) > 0)

def test_target_get_model():
    state = State(experiment="test")
    state.create_target("test-target")
    state.create_model("test-model", "test-target")
    target = state.get_target("test-target")
    model = state.get_model("test-model", "test-target")
    assert(model == target["test-model"])
