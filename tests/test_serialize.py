from pathlib import Path
import json

from smartsim import Experiment
from smartsim._core.utils import serialize
from smartsim._core.control.manifest import LaunchedManifestBuilder


def test_serialize_creates_a_manifest_json_file_if_dne(fileutils):
    test_dir = fileutils.get_test_dir()
    lmb = LaunchedManifestBuilder()
    serialize.save_launch_manifest(lmb.finalize("exp", test_dir, "launcher"))
    manifest_json = Path(test_dir) / ".smartsim/manifest/manifest.json"

    assert manifest_json.is_file()
    with open(manifest_json, 'r') as f:
        manifest = json.load(f)
        assert manifest["experiment"]["name"] == "exp"
        assert manifest["experiment"]["launcher"] == "launcher"
        assert isinstance(manifest["runs"], list)
        assert len(manifest["runs"]) == 1


def test_serialize_appends_a_manifest_json_exists(fileutils):
    test_dir = fileutils.get_test_dir()
    lmb = LaunchedManifestBuilder()
    serialize.save_launch_manifest(lmb.finalize("exp", test_dir, "launcher"))
    manifest_json = Path(test_dir) / ".smartsim/manifest/manifest.json"
    serialize.save_launch_manifest(lmb.finalize("exp", test_dir, "launcher"))
    serialize.save_launch_manifest(lmb.finalize("exp", test_dir, "launcher"))

    assert manifest_json.is_file()
    with open(manifest_json, 'r') as f:
        manifest = json.load(f)
        assert isinstance(manifest["runs"], list)
        assert len(manifest["runs"]) == 3


def test_serialize_overwites_file_if_not_json(fileutils):
    test_dir = fileutils.get_test_dir()
    manifest_json = Path(test_dir) / ".smartsim/manifest/manifest.json"
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_json, 'w') as f:
        f.write("This is not a json\n")

    lmb = LaunchedManifestBuilder()
    serialize.save_launch_manifest(lmb.finalize("exp", test_dir, "launcher"))
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

    manifest_json = Path(exp.exp_path) / ".smartsim/manifest/manifest.json"
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
