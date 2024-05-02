from smartsim.entity import Ensemble, Model
from smartsim.settings import RunSettings
from smartsim.database import Orchestrator
from smartsim import Experiment
from smartsim.status import SmartSimStatus

def test_model_constructor():
    run_settings = RunSettings()
    model = Model(name="testing", run_settings=run_settings, exe="echo", exe_args=["hello"], params={})
    assert model.exe == ["/bin/echo"]
    assert model.exe_args == ["hello"]

def test_model_add_exe_args():
    run_settings = RunSettings()
    model = Model(name="testing", run_settings=run_settings, exe="echo", exe_args=["hello"], params={})
    model.add_exe_args("there")
    assert model.exe_args == ["hello", "there"]
    model.add_exe_args(["how", "are", "you"])
    assert model.exe_args == ["hello", "there", "how", "are", "you"]

def test_create_model():
    run_settings = RunSettings()
    exp = Experiment("exp")
    model = exp.create_model(name="model", run_settings=run_settings, exe="echo", exe_args=["hello"])
    assert model.exe == ["/bin/echo"]
    assert model.exe_args == ["hello"]

def test_start_a_model():
    exp = Experiment("exp")
    run_settings = RunSettings()
    model = Model(name="testing", exe="echo", run_settings=run_settings, exe_args=["hello"], params={})
    assert model.exe == ["/bin/echo"]
    assert model.exe_args == ["hello"]
    exp.start(model)
    # if model failed, test will fail
    model_status = exp.get_status(model)[0]
    assert model_status != SmartSimStatus.STATUS_FAILED

# def test_ensemble_constructor():
#     ensemble = Ensemble(name="testing", exe="echo", exe_args=["hello"], replicas=2, params={})
#     assert ensemble.exe == "echo"
#     assert ensemble.exe_args == ["hello"]
#     for model in ensemble:
#         assert model.exe == ["/bin/echo"]
#         assert model.exe_args == ["hello"]

# def test_ensemble_constructor():
#     ensemble = Ensemble(name="testing", exe="echo", exe_args=["hello"], perm_strat="all_perm", params= {"h": "6", "g": "8"})
#     assert ensemble.exe == "echo"
#     assert ensemble.exe_args == ["hello"]
#     for model in ensemble:
#         assert model.exe == ["/bin/echo"]
#         assert model.exe_args == ["hello"]

# def test_create_ensemble():
#     exp = Experiment("exp")
#     ensemble = exp.create_ensemble(name="model", exe="echo", exe_args=["hello"], replicas=2)
#     assert ensemble.exe == "echo"
#     assert ensemble.exe_args == ["hello"]
#     for model in ensemble:
#         assert model.exe == ["/bin/echo"]
#         assert model.exe_args == ["hello"]

# def test_orchestrator_constructor():
#     orch = Orchestrator()
#     print(f"entities: {orch.entities[0].exe}")