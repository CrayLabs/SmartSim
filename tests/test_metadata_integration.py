"""Integration tests for metadata directory functionality end-to-end"""

import pathlib
import time

from smartsim import Experiment
from smartsim._core.config import CONFIG


def _metadata_dir(exp_path: str) -> pathlib.Path:
    return pathlib.Path(exp_path) / CONFIG.metadata_subdir


def _single_run_dir(metadata_dir: pathlib.Path) -> pathlib.Path:
    run_dirs = [
        d for d in metadata_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]
    assert (
        len(run_dirs) == 1
    ), f"Should have exactly one run directory, found: {run_dirs}"
    return run_dirs[0]


def test_metadata_structure_model_only(test_dir: str) -> None:
    exp = Experiment("test_metadata_model", exp_path=test_dir, launcher="local")
    model = exp.create_model(
        "test_model", run_settings=exp.create_run_settings("echo", ["hello"])
    )

    exp.start(model, block=True)

    metadata_dir = _metadata_dir(test_dir)
    assert metadata_dir.is_dir(), "Metadata directory should exist"

    run_dir = _single_run_dir(metadata_dir)
    model_dir = run_dir / "model" / "test_model"
    ensemble_dir = run_dir / "ensemble"
    database_dir = run_dir / "database"

    assert model_dir.is_dir(), f"Model metadata directory should exist: {model_dir}"
    assert not ensemble_dir.exists(), f"Unexpected ensemble directory: {ensemble_dir}"
    assert not database_dir.exists(), f"Unexpected database directory: {database_dir}"


def test_metadata_structure_ensemble_only(test_dir: str) -> None:
    exp = Experiment("test_metadata_ensemble", exp_path=test_dir, launcher="local")
    ensemble = exp.create_ensemble(
        "test_ensemble",
        run_settings=exp.create_run_settings("echo", ["world"]),
        replicas=2,
    )

    exp.start(ensemble, block=True)

    metadata_dir = _metadata_dir(test_dir)
    assert metadata_dir.is_dir(), "Metadata directory should exist"

    run_dir = _single_run_dir(metadata_dir)
    model_dir = run_dir / "model"
    ensemble_dir = run_dir / "ensemble" / "test_ensemble"
    database_dir = run_dir / "database"

    assert not model_dir.exists(), f"Unexpected model directory: {model_dir}"
    assert ensemble_dir.is_dir(), f"Missing ensemble directory: {ensemble_dir}"
    assert not database_dir.exists(), f"Unexpected database directory: {database_dir}"


def test_metadata_structure_all_entity_types(test_dir: str) -> None:
    exp = Experiment("test_metadata_all", exp_path=test_dir, launcher="local")
    model = exp.create_model(
        "test_model", run_settings=exp.create_run_settings("echo", ["hello"])
    )
    ensemble = exp.create_ensemble(
        "test_ensemble",
        run_settings=exp.create_run_settings("echo", ["world"]),
        replicas=2,
    )
    db = exp.create_database(interface="lo")

    exp.generate(db)
    exp.start(db, model, ensemble, block=True)
    exp.stop(db)

    metadata_dir = _metadata_dir(test_dir)
    assert metadata_dir.is_dir(), "Metadata directory should exist"

    run_dir = _single_run_dir(metadata_dir)
    model_dir = run_dir / "model" / "test_model"
    ensemble_dir = run_dir / "ensemble" / "test_ensemble"
    database_dir = run_dir / "database" / db.name

    assert model_dir.is_dir(), f"Model metadata directory should exist: {model_dir}"
    assert ensemble_dir.is_dir(), f"Ensemble metadata directory should exist: {ensemble_dir}"
    assert database_dir.is_dir(), f"Database metadata directory should exist: {database_dir}"


def test_multiple_runs_create_unique_directories(test_dir: str) -> None:
    exp1 = Experiment("test_metadata_run1", exp_path=test_dir, launcher="local")
    model1 = exp1.create_model(
        "test_model1", run_settings=exp1.create_run_settings("echo", ["run1"])
    )
    exp1.start(model1, block=True)

    time.sleep(0.01)

    exp2 = Experiment("test_metadata_run2", exp_path=test_dir, launcher="local")
    model2 = exp2.create_model(
        "test_model2", run_settings=exp2.create_run_settings("echo", ["run2"])
    )
    exp2.start(model2, block=True)

    metadata_dir = _metadata_dir(test_dir)
    run_dirs = [
        d for d in metadata_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]
    assert len(run_dirs) == 2, f"Should have exactly two run directories, found: {run_dirs}"

    expected_models = {"test_model1", "test_model2"}
    discovered = set()
    for run_dir in run_dirs:
        model_parent = run_dir / "model"
        assert model_parent.is_dir(), f"Model directory missing in {run_dir}"
        matches = [name for name in expected_models if (model_parent / name).exists()]
        assert matches, f"No model directory found in {run_dir}"
        discovered.add(matches[0])

    assert discovered == expected_models, f"Model directories mismatch: {discovered}"


def test_metadata_directory_permissions(test_dir: str) -> None:
    exp = Experiment("test_metadata_perms", exp_path=test_dir, launcher="local")
    model = exp.create_model(
        "test_model", run_settings=exp.create_run_settings("echo", ["permissions"])
    )

    exp.start(model, block=True)

    metadata_dir = _metadata_dir(test_dir)
    assert metadata_dir.is_dir(), "Metadata directory should exist"
    assert metadata_dir.stat().st_mode & 0o700

    run_dir = _single_run_dir(metadata_dir)
    model_dir = run_dir / "model" / "test_model"
    assert model_dir.is_dir(), f"Model metadata directory should exist: {model_dir}"
    assert model_dir.stat().st_mode & 0o700
