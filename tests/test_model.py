import pytest

from smartsim import Experiment
from smartsim.error import EntityExistsError
from smartsim.settings import RunSettings


def test_register_incoming_entity_preexists():
    exp = Experiment("experiment", launcher="local")
    rs = RunSettings("python", exe_args="sleep.py")
    ensemble = exp.create_ensemble(name="ensemble", replicas=1, run_settings=rs)
    m = exp.create_model("model", run_settings=rs)
    m.register_incoming_entity(ensemble[0])
    assert len(m.incoming_entities) == 1
    with pytest.raises(EntityExistsError):
        m.register_incoming_entity(ensemble[0])


def test_disable_key_prefixing():
    exp = Experiment("experiment", launcher="local")
    rs = RunSettings("python", exe_args="sleep.py")
    m = exp.create_model("model", run_settings=rs)
    m.disable_key_prefixing()
    assert m.query_key_prefixing() == False


def test_repr():
    expr = Experiment("experiment")
    m = expr.create_model("test_model", run_settings=RunSettings("python"))
    assert m.__repr__() == "test_model"


def test_str():
    expr = Experiment("experiment")
    rs = RunSettings("python", exe_args="sleep.py")
    m = expr.create_model("test_model", run_settings=rs)
    entity_str = "Name: " + m.name + "\nType: " + m.type + "\n" + str(rs)
    assert m.__str__() == entity_str


# def test_str(fileutils):
#     exp_name = "test_model_str"
#     expr = Experiment(exp_name)
#     test_dir = fileutils.make_test_dir(exp_name)
#     script = fileutils.get_test_conf_path("sleep.py")
#     rs = RunSettings("python", script)
#     m = expr.create_model("test_model", path=test_dir, run_settings=rs)
#     entity_str = "Name: " + m.name + "\nType: " + m.type + "\n" + str(rs)
#     assert m.__str__() == entity_str
