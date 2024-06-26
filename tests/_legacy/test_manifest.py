# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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
from smartsim._core.control.manifest import (
    LaunchedManifest,
    LaunchedManifestBuilder,
    Manifest,
)
from smartsim._core.control.manifest import (
    _LaunchedManifestMetadata as LaunchedManifestMetadata,
)
from smartsim.database import FeatureStore
from smartsim.entity.dbobject import FSModel, FSScript
from smartsim.error import SmartSimError
from smartsim.settings import RunSettings

if t.TYPE_CHECKING:
    from smartsim._core.launcher.step import Step
    from smartsim.entity import Ensemble, Model

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


# ---- create entities for testing --------

_EntityResult = t.Tuple[
    Experiment, t.Tuple["Model", "Model"], "Ensemble", FeatureStore, FSModel, FSScript
]


@pytest.fixture
def entities(test_dir: str) -> _EntityResult:
    rs = RunSettings("python", "sleep.py")

    exp = Experiment("util-test", launcher="local", exp_path=test_dir)
    model = exp.create_model("model_1", run_settings=rs)
    model_2 = exp.create_model("model_1", run_settings=rs)
    ensemble = exp.create_ensemble("ensemble", run_settings=rs, replicas=1)

    orc = FeatureStore()
    orc_1 = deepcopy(orc)
    orc_1.name = "orc2"

    db_script = FSScript("some-script", "def main():\n    print('hello world')\n")
    db_model = FSModel("some-model", "TORCH", b"some-model-bytes")

    return exp, (model, model_2), ensemble, orc, db_model, db_script


def test_separate(entities: _EntityResult) -> None:
    _, (model, _), ensemble, orc, _, _ = entities

    manifest = Manifest(model, ensemble, orc)
    assert manifest.models[0] == model
    assert len(manifest.models) == 1
    assert manifest.ensembles[0] == ensemble
    assert len(manifest.ensembles) == 1
    assert manifest.fss[0] == feature_store


def test_separate_type() -> None:
    with pytest.raises(TypeError):
        _ = Manifest([1, 2, 3])  # type: ignore


def test_name_collision(entities: _EntityResult) -> None:
    _, (model, model_2), _, _, _, _ = entities

    with pytest.raises(SmartSimError):
        _ = Manifest(application, application_2)


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
    "target_obj, target_prop, target_value, has_fs_objects",
    [
        pytest.param(None, None, None, False, id="No FS Objects"),
        pytest.param("a0", "fsm", "fsm", True, id="Model w/ FS Model"),
        pytest.param("a0", "fss", "fss", True, id="Model w/ FS Script"),
        pytest.param("ens", "fsm", "fsm", True, id="Ensemble w/ FS Model"),
        pytest.param("ens", "fss", "fss", True, id="Ensemble w/ FS Script"),
        pytest.param("ens_0", "fsm", "fsm", True, id="Ensemble Member w/ FS Model"),
        pytest.param("ens_0", "fss", "fss", True, id="Ensemble Member w/ FS Script"),
    ],
)
def test_manifest_detects_fs_objects(
    monkeypatch: pytest.MonkeyPatch,
    target_obj: str,
    target_prop: str,
    target_value: str,
    has_fs_objects: bool,
    entities: _EntityResult,
) -> None:
    _, (app, _), ensemble, _, fs_model, fs_script = entities
    target_map = {
        "a0": app,
        "fsm": fs_model,
        "fss": fs_script,
        "ens": ensemble,
        "ens_0": ensemble.entities[0],
    }
    prop_map = {
        "fsm": "_fs_models",
        "fss": "_fs_scripts",
    }
    if target_obj:
        patch = (
            target_map[target_obj],
            prop_map[target_prop],
            [target_map[target_value]],
        )
        monkeypatch.setattr(*patch)

    assert Manifest(model, ensemble).has_fs_objects == has_fs_objects


def test_launched_manifest_transform_data(entities: _EntityResult) -> None:
    _, (application, application_2), ensemble, feature_store, _, _ = entities

    applications = [(application, 1), (application_2, 2)]
    ensembles = [(ensemble, [(m, i) for i, m in enumerate(ensemble.entities)])]
    fss = [(feature_store, [(n, i) for i, n in enumerate(feature_store.entities)])]
    launched = LaunchedManifest(
        metadata=LaunchedManifestMetadata("name", "path", "launcher", "run_id"),
        applications=applications,  # type: ignore
        ensembles=ensembles,  # type: ignore
        featurestores=fss,  # type: ignore
    )
    transformed = launched.map(lambda x: str(x))

    assert transformed.applications == tuple((m, str(i)) for m, i in applications)
    assert transformed.ensembles[0][1] == tuple((m, str(i)) for m, i in ensembles[0][1])
    assert transformed.featurestores[0][1] == tuple((n, str(i)) for n, i in fss[0][1])


def test_launched_manifest_builder_correctly_maps_data(entities: _EntityResult) -> None:
    _, (application, application_2), ensemble, feature_store, _, _ = entities

    lmb = LaunchedManifestBuilder(
        "name", "path", "launcher name", str(uuid4())
    )  # type: ignore
    lmb.add_application(application, 1)
    lmb.add_application(application_2, 1)
    lmb.add_ensemble(ensemble, [i for i in range(len(ensemble.entities))])
    lmb.add_feature_store(
        feature_store, [i for i in range(len(feature_store.entities))]
    )

    manifest = lmb.finalize()
    assert len(manifest.applications) == 2
    assert len(manifest.ensembles) == 1
    assert len(manifest.featurestores) == 1


def test_launced_manifest_builder_raises_if_lens_do_not_match(
    entities: _EntityResult,
) -> None:
    _, _, ensemble, orc, _, _ = entities

    lmb = LaunchedManifestBuilder(
        "name", "path", "launcher name", str(uuid4())
    )  # type: ignore
    with pytest.raises(ValueError):
        lmb.add_ensemble(ensemble, list(range(123)))
    with pytest.raises(ValueError):
        lmb.add_feature_store(feature_store, list(range(123)))


def test_launched_manifest_builer_raises_if_attaching_data_to_empty_collection(
    monkeypatch: pytest.MonkeyPatch, entities: _EntityResult
) -> None:
    _, _, ensemble, _, _, _ = entities

    lmb: LaunchedManifestBuilder[t.Tuple[str, "Step"]] = LaunchedManifestBuilder(
        "name", "path", "launcher", str(uuid4())
    )
    monkeypatch.setattr(ensemble, "entities", [])
    with pytest.raises(ValueError):
        lmb.add_ensemble(ensemble, [])


def test_lmb_and_launched_manifest_have_same_paths_for_launched_metadata() -> None:
    exp_path = "/path/to/some/exp"
    lmb: LaunchedManifestBuilder[t.Tuple[str, "Step"]] = LaunchedManifestBuilder(
        "exp_name", exp_path, "launcher", str(uuid4())
    )
    manifest = lmb.finalize()
    assert (
        lmb.exp_telemetry_subdirectory == manifest.metadata.exp_telemetry_subdirectory
    )
    assert (
        lmb.run_telemetry_subdirectory == manifest.metadata.run_telemetry_subdirectory
    )
    assert (
        os.path.commonprefix(
            [
                manifest.metadata.run_telemetry_subdirectory,
                manifest.metadata.exp_telemetry_subdirectory,
                manifest.metadata.manifest_file_path,
                exp_path,
            ]
        )
        == exp_path
    )
    assert os.path.commonprefix(
        [
            manifest.metadata.run_telemetry_subdirectory,
            manifest.metadata.exp_telemetry_subdirectory,
            manifest.metadata.manifest_file_path,
        ]
    ) == str(manifest.metadata.exp_telemetry_subdirectory)
