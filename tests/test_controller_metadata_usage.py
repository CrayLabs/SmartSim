"""Unit tests for SmartSim controller metadata handling."""

from __future__ import annotations

import os
import pathlib

import pytest

from smartsim._core.control.controller import Controller
from smartsim._core.control.manifest import Manifest
from smartsim.entity import Model
from smartsim.settings import RunSettings


class _DummyStep:
    def __init__(self, metadata_dir: pathlib.Path, entity_name: str) -> None:
        self.name = f"{entity_name}-step"
        self.entity_name = entity_name
        self.meta = {"metadata_dir": str(metadata_dir)}

    def get_output_files(self) -> tuple[str, str]:
        metadata_dir = pathlib.Path(self.meta["metadata_dir"])
        metadata_dir.mkdir(parents=True, exist_ok=True)
        out_file = metadata_dir / f"{self.entity_name}.out"
        err_file = metadata_dir / f"{self.entity_name}.err"
        out_file.touch(exist_ok=True)
        err_file.touch(exist_ok=True)
        return str(out_file), str(err_file)


def test_controller_uses_run_prefixed_metadata_dir(tmp_path, monkeypatch):
    controller = Controller("local")
    controller._jobs.get_db_host_addresses = lambda: {}
    controller._jobs.actively_monitoring = True

    recorded_dirs: list[pathlib.Path] = []

    def fake_create_job_step(entity: Model, metadata_dir: pathlib.Path) -> _DummyStep:
        recorded_dirs.append(pathlib.Path(metadata_dir))
        return _DummyStep(metadata_dir, entity.name)

    monkeypatch.setattr(controller, "_create_job_step", fake_create_job_step)
    monkeypatch.setattr(controller, "_launch_step", lambda step, entity: None)
    monkeypatch.setattr(
        controller, "symlink_output_files", lambda *args, **kwargs: None
    )

    run_settings = RunSettings("echo", ["hello"])
    model_path = tmp_path / "simple_model"
    model_path.mkdir()
    model = Model("simple_model", {}, str(model_path), run_settings)
    manifest = Manifest(model)

    controller._launch("exp", str(tmp_path), manifest)

    assert recorded_dirs, "Controller did not attempt to build metadata directories"
    metadata_dir = recorded_dirs[0]
    expected_glob = tmp_path / ".smartsim" / "metadata" / "run_*" / "model" / model.name
    assert metadata_dir.match(str(expected_glob))


def test_symlink_output_files_targets_metadata_dir(tmp_path):
    metadata_dir = tmp_path / ".smartsim" / "metadata" / "run_test" / "model" / "sample"
    metadata_dir.mkdir(parents=True)
    out_file = metadata_dir / "sample.out"
    err_file = metadata_dir / "sample.err"
    out_file.write_text("stdout")
    err_file.write_text("stderr")

    step = _DummyStep(metadata_dir, "sample")

    entity_path = tmp_path / "workdir"
    entity_path.mkdir()
    entity = type("Entity", (), {"name": "sample", "path": str(entity_path)})

    Controller.symlink_output_files(step, entity)

    entity_out = entity_path / "sample.out"
    entity_err = entity_path / "sample.err"

    assert entity_out.is_symlink()
    assert entity_err.is_symlink()
    assert os.readlink(entity_out) == str(out_file)
    assert os.readlink(entity_err) == str(err_file)
