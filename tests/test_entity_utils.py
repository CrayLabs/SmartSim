from copy import deepcopy

import pytest

from smartsim import Experiment
from smartsim.error import SmartSimError
from smartsim.utils.entityutils import separate_entities
from smartsim.database import Orchestrator
from smartsim.settings import RunSettings

# ---- create entities for testing --------

rs = RunSettings("python", "sleep.py")

exp = Experiment("util-test", launcher="local")
model = exp.create_model("model_1", run_settings=rs)
model_2 = exp.create_model("model_1", run_settings=rs)
ensemble = exp.create_ensemble("ensemble", run_settings=rs, replicas=1)
orc = Orchestrator()
orc_1 = deepcopy(orc)


def test_separate():
    ent, ent_list, _orc = separate_entities([model, ensemble, orc])
    assert ent[0] == model
    assert ent_list[0] == ensemble
    assert _orc == orc


def test_two_orc():
    with pytest.raises(SmartSimError):
        _, _, _orc = separate_entities([orc, orc_1])


def test_separate_type():
    with pytest.raises(TypeError):
        _, _, _ = separate_entities([1, 2, 3])


def test_name_collision():
    with pytest.raises(SmartSimError):
        _, _, _ = separate_entities([model, model_2])


def test_corner_case():
    """tricky corner case where some variable may have a
    name attribute
    """

    class Person:
        name = "hello"

    p = Person()
    with pytest.raises(TypeError):
        _, _, _ = separate_entities([p])
