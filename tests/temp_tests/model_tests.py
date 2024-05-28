from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.entity import Ensemble, Model
from smartsim.settings import RunSettings, SrunSettings
from smartsim.status import SmartSimStatus


def test_model_constructor():
    run_settings = RunSettings()
    model = Model(
        name="testing",
        run_settings=run_settings,
        exe="echo",
        exe_args=["hello"],
        params={},
    )
    assert model.exe == ["/usr/bin/echo"]
    assert model.exe_args == ["hello"]


def test_model_add_exe_args():
    run_settings = SrunSettings()
    model = Model(
        name="testing",
        run_settings=run_settings,
        exe="echo",
        exe_args=["hello"],
        params={},
    )
    model.add_exe_args("there")
    assert model.exe_args == ["hello", "there"]
    model.add_exe_args(["how", "are", "you"])
    assert model.exe_args == ["hello", "there", "how", "are", "you"]


def test_create_model():
    run_settings = SrunSettings()
    exp = Experiment("exp")
    model = exp.create_model(
        name="model", run_settings=run_settings, exe="echo", exe_args=["hello"]
    )
    assert model.exe == ["/usr/bin/echo"]
    assert model.exe_args == ["hello"]


def test_start_a_model():
    exp = Experiment("exp")
    run_settings = SrunSettings()
    model = Model(
        name="testing",
        exe="echo",
        run_settings=run_settings,
        exe_args=["hello"],
        params={},
    )
    assert model.exe == ["/usr/bin/echo"]
    assert model.exe_args == ["hello"]
    exp.start(model)
    model_status = exp.get_status(model)[0]
    assert model_status != SmartSimStatus.STATUS_FAILED
