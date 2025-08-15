# BSD 2-Clause License
#
# Copyright (c) 2021-2025, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os.path
import typing as t
from copy import deepcopy
from uuid import uuid4

import pytest

from smartsim import Experiment
from smartsim._core.control.manifest import Manifest
from smartsim._core.launcher.step import Step
from smartsim.database import Orchestrator
from smartsim.entity import Ensemble, Model
from smartsim.entity.dbobject import DBModel, DBScript
from smartsim.error import SmartSimError
from smartsim.settings import RunSettings

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


# ---- create entities for testing --------

_EntityResult = t.Tuple[
    Experiment, t.Tuple[Model, Model], Ensemble, Orchestrator, DBModel, DBScript
]


@pytest.fixture
def entities(test_dir: str) -> _EntityResult:
    rs = RunSettings("python", "sleep.py")

    exp = Experiment("util-test", launcher="local", exp_path=test_dir)
    model = exp.create_model("model_1", run_settings=rs)
    model_2 = exp.create_model("model_1", run_settings=rs)
    ensemble = exp.create_ensemble("ensemble", run_settings=rs, replicas=1)

    orc = Orchestrator()
    orc_1 = deepcopy(orc)
    orc_1.name = "orc2"

    db_script = DBScript("some-script", "def main():\n    print('hello world')\n")
    db_model = DBModel("some-model", "TORCH", b"some-model-bytes")

    return exp, (model, model_2), ensemble, orc, db_model, db_script


def test_separate(entities: _EntityResult) -> None:
    _, (model, _), ensemble, orc, _, _ = entities

    manifest = Manifest(model, ensemble, orc)
    assert manifest.models[0] == model
    assert len(manifest.models) == 1
    assert manifest.ensembles[0] == ensemble
    assert len(manifest.ensembles) == 1
    assert manifest.dbs[0] == orc


def test_separate_type() -> None:
    with pytest.raises(TypeError):
        _ = Manifest([1, 2, 3])  # type: ignore


def test_name_collision(entities: _EntityResult) -> None:
    _, (model, model_2), _, _, _, _ = entities

    with pytest.raises(SmartSimError):
        _ = Manifest(model, model_2)


def test_catch_empty_ensemble(entities: _EntityResult) -> None:
    _, _, ensemble, _, _, _ = entities

    e = deepcopy(ensemble)
    e.entities = []
    with pytest.raises(ValueError):
        _ = Manifest(e)


def test_corner_case() -> None:
    """tricky corner case where some variable may have a
    name attribute
    """

    class Person:
        name = "hello"

    p = Person()
    with pytest.raises(TypeError):
        _ = Manifest(p)  # type: ignore


@pytest.mark.parametrize(
    "target_obj, target_prop, target_value, has_db_objects",
    [
        pytest.param(None, None, None, False, id="No DB Objects"),
        pytest.param("m0", "dbm", "dbm", True, id="Model w/ DB Model"),
        pytest.param("m0", "dbs", "dbs", True, id="Model w/ DB Script"),
        pytest.param("ens", "dbm", "dbm", True, id="Ensemble w/ DB Model"),
        pytest.param("ens", "dbs", "dbs", True, id="Ensemble w/ DB Script"),
        pytest.param("ens_0", "dbm", "dbm", True, id="Ensemble Member w/ DB Model"),
        pytest.param("ens_0", "dbs", "dbs", True, id="Ensemble Member w/ DB Script"),
    ],
)
def test_manifest_detects_db_objects(
    monkeypatch: pytest.MonkeyPatch,
    target_obj: str,
    target_prop: str,
    target_value: str,
    has_db_objects: bool,
    entities: _EntityResult,
) -> None:
    _, (model, _), ensemble, _, db_model, db_script = entities
    target_map = {
        "m0": model,
        "dbm": db_model,
        "dbs": db_script,
        "ens": ensemble,
        "ens_0": ensemble.entities[0],
    }
    prop_map = {
        "dbm": "_db_models",
        "dbs": "_db_scripts",
    }
    if target_obj:
        patch = (
            target_map[target_obj],
            prop_map[target_prop],
            [target_map[target_value]],
        )
        monkeypatch.setattr(*patch)

    assert Manifest(model, ensemble).has_db_objects == has_db_objects


