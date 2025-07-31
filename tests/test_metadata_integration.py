"""Integration tests for metadata directory functionality end-to-end"""

import tempfile
import pathlib
import time
from unittest.mock import patch

import pytest

from smartsim import Experiment
from smartsim.entity import Model, Ensemble
from smartsim.database.orchestrator import Orchestrator
from smartsim.settings import RunSettings


class TestMetadataDirectoryIntegration:
    """Integration tests for metadata directory creation across the SmartSim workflow"""

    def test_experiment_creates_correct_metadata_directory_structure_model_only(self):
        """Test that launching only models creates the correct directory structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exp = Experiment("test_metadata_model", exp_path=temp_dir, launcher="local")

            # Create a simple model
            model = exp.create_model(
                "test_model",
                run_settings=exp.create_run_settings("echo", ["hello"])
            )

            # Start and wait for completion
            exp.start(model, block=False)
            exp.poll(interval=1)

            # Verify directory structure
            smartsim_dir = pathlib.Path(temp_dir) / ".smartsim"
            metadata_dir = smartsim_dir / "metadata"

            assert metadata_dir.exists(), "Metadata directory should exist"

            # Check for run-specific subdirectory
            run_dirs = [d for d in metadata_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            assert len(run_dirs) == 1, f"Should have exactly one run directory, found: {run_dirs}"

            run_dir = run_dirs[0]

            # Check for entity-specific subdirectories
            model_dir = run_dir / "model"
            ensemble_dir = run_dir / "ensemble"
            database_dir = run_dir / "database"

            assert model_dir.exists(), f"Model metadata directory should exist: {model_dir}"
            assert not ensemble_dir.exists(), f"Ensemble metadata directory should not exist: {ensemble_dir}"
            assert not database_dir.exists(), f"Database metadata directory should not exist: {database_dir}"

            # Clean up
            exp.stop(model)

    def test_experiment_creates_correct_metadata_directory_structure_ensemble_only(self):
        """Test that launching only ensembles creates the correct directory structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exp = Experiment("test_metadata_ensemble", exp_path=temp_dir, launcher="local")

            # Create an ensemble
            ensemble = exp.create_ensemble(
                "test_ensemble",
                run_settings=exp.create_run_settings("echo", ["world"]),
                replicas=2
            )

            # Start and wait for completion
            exp.start(ensemble, block=False)
            exp.poll(interval=1)

            # Verify directory structure
            smartsim_dir = pathlib.Path(temp_dir) / ".smartsim"
            metadata_dir = smartsim_dir / "metadata"

            assert metadata_dir.exists(), "Metadata directory should exist"

            # Check for run-specific subdirectory
            run_dirs = [d for d in metadata_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            assert len(run_dirs) == 1, f"Should have exactly one run directory, found: {run_dirs}"

            run_dir = run_dirs[0]

            # Check for entity-specific subdirectories
            model_dir = run_dir / "model"
            ensemble_dir = run_dir / "ensemble"
            database_dir = run_dir / "database"

            assert not model_dir.exists(), f"Model metadata directory should not exist: {model_dir}"
            assert ensemble_dir.exists(), f"Ensemble metadata directory should exist: {ensemble_dir}"
            assert not database_dir.exists(), f"Database metadata directory should not exist: {database_dir}"

            # Clean up
            exp.stop(ensemble)

    def test_experiment_creates_correct_metadata_directory_structure_all_types(self):
        """Test that launching models, ensembles, and orchestrator creates all directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exp = Experiment("test_metadata_all", exp_path=temp_dir, launcher="local")

            # Create model
            model = exp.create_model(
                "test_model",
                run_settings=exp.create_run_settings("echo", ["hello"])
            )

            # Create ensemble
            ensemble = exp.create_ensemble(
                "test_ensemble",
                run_settings=exp.create_run_settings("echo", ["world"]),
                replicas=2
            )

            # Create database
            orchestrator = exp.create_database(port=6379, interface="lo")

            # Start all entities - orchestrator and compute entities may create separate run dirs
            exp.start(orchestrator, block=False)
            exp.start(model, ensemble, block=False)
            exp.poll(interval=1)

            # Verify directory structure
            smartsim_dir = pathlib.Path(temp_dir) / ".smartsim"
            metadata_dir = smartsim_dir / "metadata"

            assert metadata_dir.exists(), "Metadata directory should exist"

            # Check for run-specific subdirectories (may be 1 or 2 depending on timing)
            run_dirs = [d for d in metadata_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            assert len(run_dirs) >= 1, f"Should have at least one run directory, found: {run_dirs}"

            # Find directory with model/ensemble subdirs
            run_dir = None
            for rd in run_dirs:
                if (rd / "model").exists() or (rd / "ensemble").exists():
                    run_dir = rd
                    break

            assert run_dir is not None, "Should find run directory with entity subdirs"

            # Check for entity-specific subdirectories
            model_dir = run_dir / "model"
            ensemble_dir = run_dir / "ensemble"

            assert model_dir.exists(), f"Model metadata directory should exist: {model_dir}"
            assert ensemble_dir.exists(), f"Ensemble metadata directory should exist: {ensemble_dir}"            # Clean up
            exp.stop(model, ensemble)
            exp.stop(orchestrator)

    def test_multiple_experiment_runs_create_separate_run_directories(self):
        """Test that multiple experiment runs create separate timestamped directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First experiment run
            exp1 = Experiment("test_metadata_run1", exp_path=temp_dir, launcher="local")
            model1 = exp1.create_model(
                "test_model1",
                run_settings=exp1.create_run_settings("echo", ["run1"])
            )

            exp1.start(model1, block=False)
            exp1.poll(interval=1)
            exp1.stop(model1)

            # Small delay to ensure different timestamps
            time.sleep(0.01)

            # Second experiment run
            exp2 = Experiment("test_metadata_run2", exp_path=temp_dir, launcher="local")
            model2 = exp2.create_model(
                "test_model2",
                run_settings=exp2.create_run_settings("echo", ["run2"])
            )

            exp2.start(model2, block=False)
            exp2.poll(interval=1)
            exp2.stop(model2)

            # Verify two separate run directories exist
            metadata_dir = pathlib.Path(temp_dir) / ".smartsim" / "metadata"
            run_dirs = [d for d in metadata_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

            assert len(run_dirs) == 2, f"Should have exactly two run directories, found: {run_dirs}"

            # Verify both have model subdirectories
            for run_dir in run_dirs:
                model_dir = run_dir / "model"
                assert model_dir.exists(), f"Model metadata directory should exist in {run_dir}"

    def test_metadata_directory_structure_with_batch_entities(self):
        """Test metadata directory creation pattern with batch-like behavior"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exp = Experiment("test_metadata_batch", exp_path=temp_dir, launcher="local")

            # Create model and ensemble (batch settings don't work with local launcher)
            model = exp.create_model(
                "batch_model",
                run_settings=exp.create_run_settings("echo", ["batch_hello"])
            )

            ensemble = exp.create_ensemble(
                "batch_ensemble",
                run_settings=exp.create_run_settings("echo", ["batch_world"]),
                replicas=2
            )

            # Start entities to trigger metadata directory creation
            exp.start(model, ensemble, block=False)
            exp.poll(interval=1)

            # Verify directory structure was created
            smartsim_dir = pathlib.Path(temp_dir) / ".smartsim"
            metadata_dir = smartsim_dir / "metadata"

            assert metadata_dir.exists(), "Metadata directory should exist"

            # Check for run-specific subdirectory
            run_dirs = [d for d in metadata_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            assert len(run_dirs) >= 1, f"Should have at least one run directory, found: {run_dirs}"

            # Check that at least one run directory has entity subdirs
            has_model_dir = any((rd / "model").exists() for rd in run_dirs)
            has_ensemble_dir = any((rd / "ensemble").exists() for rd in run_dirs)

            assert has_model_dir, "Should have model metadata directory"
            assert has_ensemble_dir, "Should have ensemble metadata directory"

            # Stop entities to clean up
            exp.stop(model, ensemble)

    def test_metadata_directory_permissions_and_structure(self):
        """Test that metadata directories are created with correct permissions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exp = Experiment("test_metadata_perms", exp_path=temp_dir, launcher="local")

            model = exp.create_model(
                "test_model",
                run_settings=exp.create_run_settings("echo", ["permissions"])
            )

            exp.start(model, block=False)
            exp.poll(interval=1)

            # Check directory structure and permissions
            smartsim_dir = pathlib.Path(temp_dir) / ".smartsim"
            metadata_dir = smartsim_dir / "metadata"

            # Verify directories exist and are readable/writable
            assert metadata_dir.exists() and metadata_dir.is_dir()
            assert metadata_dir.stat().st_mode & 0o700  # Owner should have read/write/execute

            run_dirs = [d for d in metadata_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            if run_dirs:
                run_dir = run_dirs[0]
                assert run_dir.exists() and run_dir.is_dir()

                model_dir = run_dir / "model"
                if model_dir.exists():
                    assert model_dir.is_dir()
                    assert model_dir.stat().st_mode & 0o700

            exp.stop(model)
