from os import environ, getcwd, path
from shutil import rmtree

import pytest

from smartsim import Experiment
from smartsim.entity import Ensemble, Model
from smartsim.error import LauncherError, SmartSimError

# --- Simple Entity Creation ----------------------------------------------

rs = {"executable": "python"}


def test_create_model():
    exp = Experiment("test")
    model = exp.create_model("test-model", run_settings=rs)
    assert type(model) == Model
    assert model.query_key_prefixing() == False


def test_create_model_in_ensemble():
    exp = Experiment("test")
    ensemble = exp.create_ensemble("test-ensemble", run_settings=rs)
    model = exp.create_model("test-model", run_settings=rs)
    ensemble.add_model(model)
    assert ensemble.entities[0] == model


def test_create_empty_ensemble():
    experiment = Experiment("test")
    ensemble = experiment.create_ensemble("test-ensemble", run_settings=rs)
    assert len(ensemble) == 0


# --- Error Handling ---------------------------------------------------


def test_duplicate_orchestrator_error():
    experiment = Experiment("test")
    experiment.create_orchestrator()

    with pytest.raises(SmartSimError):
        experiment.create_orchestrator()


def test_invalid_num_orc():
    """test creating an orchestrator with 2 nodes"""
    experiment = Experiment("test")
    with pytest.raises(SmartSimError):
        experiment.create_orchestrator(db_nodes=2)


# --- Local Launcher Experiment -------------------------------------


def test_bad_cluster_orc():
    """test when a user creates an experiment with a local launcher
    but requests a clusted orchestrator"""
    experiment = Experiment("test", launcher="local")
    with pytest.raises(SmartSimError):
        experiment.create_orchestrator(db_nodes=3)
