from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.entity import Application, Ensemble
from smartsim.settings import RunSettings, SrunSettings
from smartsim.status import SmartSimStatus


def test_application_constructor():
    run_settings = RunSettings()
    application = Application(
        name="testing",
        run_settings=run_settings,
        exe="echo",
        exe_args=["hello"],
        params={},
    )
    assert application.exe == ["/usr/bin/echo"]
    assert application.exe_args == ["hello"]


def test_application_add_exe_args():
    run_settings = SrunSettings()
    application = Application(
        name="testing",
        run_settings=run_settings,
        exe="echo",
        exe_args=["hello"],
        params={},
    )
    application.add_exe_args("there")
    assert application.exe_args == ["hello", "there"]
    application.add_exe_args(["how", "are", "you"])
    assert application.exe_args == ["hello", "there", "how", "are", "you"]


def test_create_application():
    run_settings = SrunSettings()
    exp = Experiment("exp")
    application = exp.create_application(
        name="application", run_settings=run_settings, exe="echo", exe_args=["hello"]
    )
    assert application.exe == ["/usr/bin/echo"]
    assert application.exe_args == ["hello"]


def test_start_a_application():
    exp = Experiment("exp")
    run_settings = SrunSettings()
    application = Application(
        name="testing",
        exe="echo",
        run_settings=run_settings,
        exe_args=["hello"],
        params={},
    )
    assert application.exe == ["/usr/bin/echo"]
    assert application.exe_args == ["hello"]
    exp.start(application)
    application_status = exp.get_status(application)[0]
    assert application_status != SmartSimStatus.STATUS_FAILED
