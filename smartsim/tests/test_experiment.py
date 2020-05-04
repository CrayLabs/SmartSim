
from smartsim import Experiment
from os import path, environ, getcwd
from shutil import rmtree
from ..error import SmartSimError, SSModelExistsError, LauncherError
from smartsim.entity import Ensemble, NumModel
import pytest


# --- Simple Entity Creation ----------------------------------------------

def test_create_model():
    experiment = Experiment("test")
    experiment.create_model("test-model")
    assert(type(experiment.ensembles[0].models["test-model"]) == NumModel)

def test_create_model_in_ensemble():
    experiment = Experiment("test")
    experiment.create_ensemble("test-ensemble")
    experiment.create_model("test-model", "test-ensemble")
    assert(type(experiment.ensembles[0].models["test-model"]) == NumModel)

def test_get_model():
    experiment = Experiment("test")
    experiment.create_ensemble("test-ensemble")
    experiment.create_model("test-model", "test-ensemble")
    assert(experiment.ensembles[0].models["test-model"] == experiment.get_model("test-model", "test-ensemble"))

def test_create_ensemble():
    experiment = Experiment("test")
    experiment.create_ensemble("test-ensemble")
    assert(len(experiment.ensembles) > 0)
    assert(type(experiment.ensembles[0]) == Ensemble)

def test_get_ensemble():
    experiment = Experiment("test")
    ensemble = experiment.create_ensemble("test-ensemble")
    assert(ensemble == experiment.ensembles[0])

def test_delete_ensemble():
    experiment = Experiment("test")
    experiment.create_ensemble("test-ensemble")
    assert(len(experiment.ensembles) > 0)
    assert(type(experiment.ensembles[0]) == Ensemble)
    experiment.delete_ensemble("test-ensemble")
    assert(len(experiment.ensembles) == 0)

def test_create_node():
    experiment = Experiment("test")
    experiment.create_node("test-node", script_path=getcwd())
    assert(len(experiment.nodes) > 0)

def test_ensemble_get_model():
    experiment = Experiment("test")
    ensemble = experiment.create_ensemble("test-ensemble")
    model = experiment.create_model("test-model", "test-ensemble")
    assert(model == ensemble["test-model"])

# --- Error Handling ---------------------------------------------------

def test_get_ensemble_error():
    experiment = Experiment("test")
    experiment.create_ensemble("test-ensemble")
    with pytest.raises(SmartSimError):
        ensemble = experiment.get_ensemble("test-ensemble_doesnt_exist")

def test_duplicate_orchestrator_error():
    experiment = Experiment("test")
    experiment.create_orchestrator()

    with pytest.raises(SmartSimError):
        experiment.create_orchestrator()

def test_remote_launch():
    """Test the setup of a remote launcher when a cmd_center has
       not been launched

       Test assumes that cmd_center is not running
    """
    experiment = Experiment("test")
    with pytest.raises(SmartSimError):
        experiment.init_remote_launcher()

def test_bad_release():
    """test when experiment.release() is called with a bad alloc_id"""
    experiment = Experiment("test")
    with pytest.raises(SmartSimError):
        experiment.release(alloc_id=111111)

def test_invalid_num_orc():
    """test creating an orchestrator with 2 nodes"""
    experiment = Experiment("test")
    with pytest.raises(SmartSimError):
        experiment.create_orchestrator_cluster(111111, db_nodes=2)

# --- Local Launcher Experiment -------------------------------------

def test_bad_cluster_orc():
    """test when a user creates an experiment with a local launcher
       but requests a clusted orchestrator"""
    experiment = Experiment("test", launcher="local")
    with pytest.raises(SmartSimError):
        experiment.create_orchestrator_cluster(111111)