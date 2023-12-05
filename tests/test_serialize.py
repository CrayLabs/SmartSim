# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import pytest

from pathlib import Path
import json

from smartsim import Experiment
from smartsim._core.utils import serialize
from smartsim._core.control.manifest import LaunchedManifestBuilder
import smartsim._core.config.config

_REL_MANIFEST_PATH = f"{serialize.TELMON_SUBDIR}/{serialize.MANIFEST_FILENAME}"
_CFG_TM_ENABLED_ATTR = "telemetry_enabled"

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b

@pytest.fixture(autouse=True)
def turn_on_tm(monkeypatch):
    monkeypatch.setattr(
        smartsim._core.config.config.Config,
        _CFG_TM_ENABLED_ATTR,
        property(lambda self: True))
    yield


def test_serialize_creates_a_manifest_json_file_if_dne(fileutils):
    test_dir = fileutils.get_test_dir()
    lmb = LaunchedManifestBuilder("exp", test_dir, "launcher")
    serialize.save_launch_manifest(lmb.finalize())
    manifest_json = Path(test_dir) / _REL_MANIFEST_PATH

    assert manifest_json.is_file()
    with open(manifest_json, 'r') as f:
        manifest = json.load(f)
        assert manifest["experiment"]["name"] == "exp"
        assert manifest["experiment"]["launcher"] == "launcher"
        assert isinstance(manifest["runs"], list)
        assert len(manifest["runs"]) == 1


def test_serialize_does_not_write_manifest_json_if_telemetry_monitor_is_off(
    fileutils, monkeypatch
):
    monkeypatch.setattr(
        smartsim._core.config.config.Config,
        _CFG_TM_ENABLED_ATTR,
        property(lambda self: False))
    test_dir = fileutils.get_test_dir()
    lmb = LaunchedManifestBuilder("exp", test_dir, "launcher")
    serialize.save_launch_manifest(lmb.finalize())
    manifest_json = Path(test_dir) / _REL_MANIFEST_PATH
    assert not manifest_json.exists()


def test_serialize_appends_a_manifest_json_exists(fileutils):
    test_dir = fileutils.get_test_dir()
    manifest_json = Path(test_dir) / _REL_MANIFEST_PATH
    serialize.save_launch_manifest(
        LaunchedManifestBuilder("exp", test_dir, "launcher").finalize())
    serialize.save_launch_manifest(
        LaunchedManifestBuilder("exp", test_dir, "launcher").finalize())
    serialize.save_launch_manifest(
        LaunchedManifestBuilder("exp", test_dir, "launcher").finalize())

    assert manifest_json.is_file()
    with open(manifest_json, 'r') as f:
        manifest = json.load(f)
        assert isinstance(manifest["runs"], list)
        assert len(manifest["runs"]) == 3
        assert len({run["run_id"] for run in manifest["runs"]}) == 3


def test_serialize_overwites_file_if_not_json(fileutils):
    test_dir = fileutils.get_test_dir()
    manifest_json = Path(test_dir) / _REL_MANIFEST_PATH
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_json, 'w') as f:
        f.write("This is not a json\n")

    lmb = LaunchedManifestBuilder("exp", test_dir, "launcher")
    serialize.save_launch_manifest(lmb.finalize())
    with open(manifest_json, 'r') as f:
        assert isinstance(json.load(f), dict)


def test_started_entities_are_serialized(fileutils):
    exp_name = "test-exp"
    test_dir = Path(fileutils.make_test_dir()) / exp_name
    test_dir.mkdir(parents=True)
    exp = Experiment(exp_name, exp_path=str(test_dir), launcher="local")

    rs1 = exp.create_run_settings("echo", ["hello", "world"])
    rs2 = exp.create_run_settings("echo", ["spam", "eggs"])

    hello_world_model = exp.create_model("echo-hello", run_settings=rs1)
    spam_eggs_model = exp.create_model("echo-spam", run_settings=rs2)
    hello_ensemble = exp.create_ensemble('echo-ensemble', run_settings=rs1, replicas=3)

    exp.generate(hello_world_model, spam_eggs_model, hello_ensemble)
    exp.start(hello_world_model, spam_eggs_model, block=False)
    exp.start(hello_ensemble, block=False)

    manifest_json = Path(exp.exp_path) / _REL_MANIFEST_PATH
    try:
        with open(manifest_json, 'r') as f:
            manifest = json.load(f)
            assert len(manifest["runs"]) == 2
            assert len(manifest["runs"][0]["model"]) == 2
            assert len(manifest["runs"][0]["ensemble"]) == 0
            assert len(manifest["runs"][1]["model"]) == 0
            assert len(manifest["runs"][1]["ensemble"]) == 1
            assert len(manifest["runs"][1]["ensemble"][0]["models"]) == 3
    finally:
        exp.stop(hello_world_model, spam_eggs_model, hello_ensemble)
