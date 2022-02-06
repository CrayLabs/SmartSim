from copy import deepcopy

import pytest

from smartsim import Experiment
from smartsim._core.control import Manifest
from smartsim.database import Orchestrator
from smartsim.error import SmartSimError
from smartsim.exp.ray import RayCluster
from smartsim.settings import RunSettings

# Ensure tensorflow is imported before ray. This is a workaround
# for a seg fault happening in the CI on Ubuntu when ray was being
# imported before tensorflow
try:
    import tensorflow
except ImportError:
    pass

ray_ok = True
try:
    import ray
except ImportError:
    ray_ok = False

# ---- create entities for testing --------

rs = RunSettings("python", "sleep.py")

exp = Experiment("util-test", launcher="local")
model = exp.create_model("model_1", run_settings=rs)
model_2 = exp.create_model("model_1", run_settings=rs)
ensemble = exp.create_ensemble("ensemble", run_settings=rs, replicas=1)


orc = Orchestrator()
orc_1 = deepcopy(orc)
orc_1.name = "orc2"
model_no_name = exp.create_model(name=None, run_settings=rs)
if ray_ok:
    rc = RayCluster(name="ray-cluster", workers=0, launcher="slurm", run_command="srun")


def test_separate():
    if ray_ok:
        manifest = Manifest(model, ensemble, orc, rc)
    else:
        manifest = Manifest(model, ensemble, orc)
    assert manifest.models[0] == model
    assert len(manifest.models) == 1
    assert manifest.ensembles[0] == ensemble
    assert len(manifest.ensembles) == 1
    assert manifest.db == orc
    if ray_ok:
        assert len(manifest.ray_clusters) == 1
        assert manifest.ray_clusters[0] == rc


def test_no_name():
    with pytest.raises(AttributeError):
        _ = Manifest(model_no_name)


def test_two_orc():
    with pytest.raises(SmartSimError):
        manifest = Manifest(orc, orc_1)
        manifest.db


def test_separate_type():
    with pytest.raises(TypeError):
        _ = Manifest([1, 2, 3])


def test_name_collision():
    with pytest.raises(SmartSimError):
        _ = Manifest(model, model_2)


def test_catch_empty_ensemble():
    e = deepcopy(ensemble)
    e.entities = []
    with pytest.raises(ValueError):
        manifest = Manifest(e)


def test_corner_case():
    """tricky corner case where some variable may have a
    name attribute
    """

    class Person:
        name = "hello"

    p = Person()
    with pytest.raises(TypeError):
        _ = Manifest(p)
