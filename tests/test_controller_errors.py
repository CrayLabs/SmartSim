import pytest

from smartsim._core.control import Controller, Manifest
from smartsim.database import Orchestrator, PBSOrchestrator
from smartsim.entity import Ensemble, Model
from smartsim.error import SmartSimError, SSConfigError, SSUnsupportedError
from smartsim.error.errors import SSConfigError, SSUnsupportedError
from smartsim.settings import RunSettings, SbatchSettings


def test_finished_entity_orc_error():
    """Orchestrators are never 'finished', either run forever or stopped by user"""
    orc = Orchestrator()
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.finished(orc)


def test_finished_entity_wrong_type():
    """Wrong type supplied to controller.finished"""
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.finished([])


def test_finished_not_found():
    """Ask if model is finished that hasnt been launched by this experiment"""
    rs = RunSettings("python")
    model = Model("hello", {}, "./", rs)
    cont = Controller(launcher="local")
    with pytest.raises(ValueError):
        cont.finished(model)


def test_entity_status_wrong_type():
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.get_entity_status([])


def test_entity_list_status_wrong_type():
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.get_entity_list_status([])


def test_unsupported_launcher():
    """Test when user provideds unsupported launcher"""
    cont = Controller(launcher="local")
    with pytest.raises(SSUnsupportedError):
        cont.init_launcher("thelauncherwhichdoesnotexist")


def test_no_launcher():
    """Test when user provideds unsupported launcher"""
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.init_launcher(None)


def test_wrong_orchestrator():
    # lo interface to avoid warning from SmartSim
    orc = PBSOrchestrator(6780, db_nodes=3, interface="lo", run_command="aprun")
    cont = Controller(launcher="local")
    manifest = Manifest(orc)
    with pytest.raises(SmartSimError):
        cont._launch(manifest)


def test_bad_orc_checkpoint():
    checkpoint = "./bad-checkpoint"
    cont = Controller(launcher="local")
    with pytest.raises(FileNotFoundError):
        cont.reload_saved_db(checkpoint)
