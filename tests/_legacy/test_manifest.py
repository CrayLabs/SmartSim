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

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


# ---- create entities for testing --------

rs = RunSettings("python", "sleep.py")

exp = Experiment("util-test", launcher="local")
application = exp.create_application("application_1", run_settings=rs)
application_2 = exp.create_application("application_1", run_settings=rs)
ensemble = exp.create_ensemble("ensemble", run_settings=rs, replicas=1)

feature_store = FeatureStore()
feature_store_1 = deepcopy(feature_store)
feature_store_1.name = "feature_store2"

fs_script = FSScript("some-script", "def main():\n    print('hello world')\n")
fs_model = FSModel("some-model", "TORCH", b"some-model-bytes")


def test_separate():
    manifest = Manifest(application, ensemble, feature_store)
    assert manifest.applications[0] == application
    assert len(manifest.applications) == 1
    assert manifest.ensembles[0] == ensemble
    assert len(manifest.ensembles) == 1
    assert manifest.fss[0] == feature_store


def test_separate_type():
    with pytest.raises(TypeError):
        _ = Manifest([1, 2, 3])


def test_name_collision():
    with pytest.raises(SmartSimError):
        _ = Manifest(application, application_2)


def test_catch_empty_ensemble():
    e = deepcopy(ensemble)
    e.entities = []
    with pytest.raises(ValueError):
        _ = Manifest(e)


def test_corner_case():
    """tricky corner case where some variable may have a
    name attribute
    """

    class Person:
        name = "hello"

    p = Person()
    with pytest.raises(TypeError):
        _ = Manifest(p)


@pytest.mark.parametrize(
    "patch, has_fs_objects",
    [
        pytest.param((), False, id="No FS Objects"),
        pytest.param(
            (application, "_fs_models", [fs_model]), True, id="Application w/ FS Model"
        ),
        pytest.param(
            (application, "_fs_scripts", [fs_script]),
            True,
            id="Application w/ FS Script",
        ),
        pytest.param(
            (ensemble, "_fs_models", [fs_model]), True, id="Ensemble w/ fs Model"
        ),
        pytest.param(
            (ensemble, "_fs_scripts", [fs_script]), True, id="Ensemble w/ fs Script"
        ),
        pytest.param(
            (ensemble.entities[0], "_fs_models", [fs_model]),
            True,
            id="Ensemble Member w/ fs Model",
        ),
        pytest.param(
            (ensemble.entities[0], "_fs_scripts", [fs_script]),
            True,
            id="Ensemble Member w/ fs Script",
        ),
    ],
)
def test_manifest_detects_fs_objects(monkeypatch, patch, has_fs_objects):
    if patch:
        monkeypatch.setattr(*patch)
    assert Manifest(application, ensemble).has_fs_objects == has_fs_objects


def test_launched_manifest_transform_data():
    applications = [(application, 1), (application_2, 2)]
    ensembles = [(ensemble, [(m, i) for i, m in enumerate(ensemble.entities)])]
    fss = [(feature_store, [(n, i) for i, n in enumerate(feature_store.entities)])]
    launched = LaunchedManifest(
        metadata=LaunchedManifestMetadata("name", "path", "launcher", "run_id"),
        applications=applications,
        ensembles=ensembles,
        featurestores=fss,
    )
    transformed = launched.map(lambda x: str(x))
    assert transformed.applications == tuple((m, str(i)) for m, i in applications)
    assert transformed.ensembles[0][1] == tuple((m, str(i)) for m, i in ensembles[0][1])
    assert transformed.featurestores[0][1] == tuple((n, str(i)) for n, i in fss[0][1])


def test_launched_manifest_builder_correctly_maps_data():
    lmb = LaunchedManifestBuilder("name", "path", "launcher name", str(uuid4()))
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


def test_launced_manifest_builder_raises_if_lens_do_not_match():
    lmb = LaunchedManifestBuilder("name", "path", "launcher name", str(uuid4()))
    with pytest.raises(ValueError):
        lmb.add_ensemble(ensemble, list(range(123)))
    with pytest.raises(ValueError):
        lmb.add_feature_store(feature_store, list(range(123)))


def test_launched_manifest_builer_raises_if_attaching_data_to_empty_collection(
    monkeypatch,
):
    lmb = LaunchedManifestBuilder("name", "path", "launcher", str(uuid4()))
    monkeypatch.setattr(ensemble, "entities", [])
    with pytest.raises(ValueError):
        lmb.add_ensemble(ensemble, [])


def test_lmb_and_launched_manifest_have_same_paths_for_launched_metadata():
    exp_path = "/path/to/some/exp"
    lmb = LaunchedManifestBuilder("exp_name", exp_path, "launcher", str(uuid4()))
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
