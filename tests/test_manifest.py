from copy import deepcopy

import pytest

from smartsim import Experiment
from smartsim.control import Manifest
from smartsim.database import Orchestrator
from smartsim.ray import RayCluster
from smartsim.error import SmartSimError
from smartsim.settings import RunSettings

# ---- create entities for testing --------

rs = RunSettings("python", "sleep.py")

exp = Experiment("util-test", launcher="local")
model = exp.create_model("model_1", run_settings=rs)
model_2 = exp.create_model("model_1", run_settings=rs)
ensemble = exp.create_ensemble("ensemble", run_settings=rs, replicas=1)
ray_cluster =  RayCluster(name="ray-cluster", workers=1, launcher='pbs')

orc = Orchestrator()
orc_1 = deepcopy(orc)
orc_1.name = "orc2"
model_no_name = exp.create_model(name=None, run_settings=rs)


def test_separate():
    manifest = Manifest(model, ensemble, orc)
    assert manifest.models[0] == model
    assert len(manifest.models) == 1
    assert manifest.ensembles[0] == ensemble
    assert len(manifest.ensembles) == 1
    assert manifest.db == orc


def test_no_name():
    with pytest.raises(AttributeError):
        manifest = Manifest(model_no_name)


def test_two_orc():
    with pytest.raises(SmartSimError):
        manifest = Manifest(orc, orc_1)
        manifest.db


def test_separate_type():
    with pytest.raises(TypeError):
        manifest = Manifest([1, 2, 3])


def test_name_collision():
    with pytest.raises(SmartSimError):
        manifest = Manifest(model, model_2)


def test_corner_case():
    """tricky corner case where some variable may have a
    name attribute
    """

    class Person:
        name = "hello"

    p = Person()
    with pytest.raises(TypeError):
        manifest = Manifest(p)
