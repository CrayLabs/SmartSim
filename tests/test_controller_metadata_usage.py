"""Test the controller's metadata directory usage patterns"""

import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from smartsim._core.control.controller import Controller
from smartsim._core.control.manifest import LaunchedManifestBuilder, Manifest
from smartsim.database import Orchestrator
from smartsim.entity import Ensemble, Model
from smartsim.settings import RunSettings


class TestControllerMetadataDirectoryUsage:
    """Test that the Controller properly uses metadata directories"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.controller = Controller("local")

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_controller_creates_base_metadata_directory(self):
        """Test that Controller creates the base metadata directory"""
        manifest = Manifest()  # Empty manifest

        with patch.object(self.controller, "_jobs") as mock_jobs:
            mock_jobs.get_db_host_addresses.return_value = {}
            mock_jobs.actively_monitoring = False

            # Mock the manifest builder's mkdir to track calls
            with patch.object(pathlib.Path, "mkdir") as mock_mkdir:
                launched_manifest = self.controller._launch(
                    "test_exp", self.temp_dir, manifest
                )

                # Verify that mkdir was called for the base metadata directory
                # The base metadata directory should be created
                mkdir_calls = [call for call in mock_mkdir.call_args_list]
                assert len(mkdir_calls) >= 1  # At least the base directory

                # Check that the call included parents=True, exist_ok=True
                base_mkdir_call = mkdir_calls[0]
                assert base_mkdir_call[1]["parents"] is True
                assert base_mkdir_call[1]["exist_ok"] is True

    def test_controller_creates_model_metadata_directory_only_when_models_present(self):
        """Test that model metadata directory is created only when models are present"""
        # Create manifest with model
        model = Model("test_model", {}, RunSettings("echo", ["hello"]))
        manifest = Manifest(model)

        with (
            patch.object(self.controller, "_jobs") as mock_jobs,
            patch.object(self.controller, "_launch_step") as mock_launch_step,
            patch.object(self.controller, "symlink_output_files") as mock_symlink,
        ):

            mock_jobs.get_db_host_addresses.return_value = {}
            mock_jobs.actively_monitoring = False

            # Track LaunchedManifestBuilder method calls
            with patch.object(
                LaunchedManifestBuilder, "get_entity_metadata_subdirectory"
            ) as mock_get_dir:
                mock_metadata_dir = MagicMock()
                mock_get_dir.return_value = mock_metadata_dir

                launched_manifest = self.controller._launch(
                    "test_exp", self.temp_dir, manifest
                )

                # Verify that get_entity_metadata_subdirectory was called for "model"
                model_calls = [
                    call
                    for call in mock_get_dir.call_args_list
                    if call[0][0] == "model"
                ]
                assert len(model_calls) == 1  # Should be called once for model

    def test_controller_creates_ensemble_metadata_directory_only_when_ensembles_present(
        self,
    ):
        """Test that ensemble metadata directory is created only when ensembles are present"""
        # Create manifest with ensemble
        run_settings = RunSettings("echo", ["world"])
        ensemble = Ensemble("test_ensemble", {}, run_settings=run_settings, replicas=2)
        manifest = Manifest(ensemble)

        with (
            patch.object(self.controller, "_jobs") as mock_jobs,
            patch.object(self.controller, "_launch_step") as mock_launch_step,
            patch.object(self.controller, "symlink_output_files") as mock_symlink,
        ):

            mock_jobs.get_db_host_addresses.return_value = {}
            mock_jobs.actively_monitoring = False

            # Track LaunchedManifestBuilder method calls
            with patch.object(
                LaunchedManifestBuilder, "get_entity_metadata_subdirectory"
            ) as mock_get_dir:
                mock_metadata_dir = MagicMock()
                mock_get_dir.return_value = mock_metadata_dir

                launched_manifest = self.controller._launch(
                    "test_exp", self.temp_dir, manifest
                )

                # Verify that get_entity_metadata_subdirectory was called for "ensemble"
                ensemble_calls = [
                    call
                    for call in mock_get_dir.call_args_list
                    if call[0][0] == "ensemble"
                ]
                assert len(ensemble_calls) == 1  # Should be called once for ensemble

    def test_controller_does_not_create_entity_dirs_for_missing_entity_types(self):
        """Test that entity metadata directories are not created for missing entity types"""
        # Create manifest with only a model (no ensemble, no database)
        model = Model("test_model", {}, RunSettings("echo", ["hello"]))
        manifest = Manifest(model)

        with (
            patch.object(self.controller, "_jobs") as mock_jobs,
            patch.object(self.controller, "_launch_step") as mock_launch_step,
            patch.object(self.controller, "symlink_output_files") as mock_symlink,
        ):

            mock_jobs.get_db_host_addresses.return_value = {}
            mock_jobs.actively_monitoring = False

            # Track LaunchedManifestBuilder method calls
            with patch.object(
                LaunchedManifestBuilder, "get_entity_metadata_subdirectory"
            ) as mock_get_dir:
                mock_metadata_dir = MagicMock()
                mock_get_dir.return_value = mock_metadata_dir

                launched_manifest = self.controller._launch(
                    "test_exp", self.temp_dir, manifest
                )

                # Only "model" should be requested, not "ensemble" or "database"
                requested_types = [call[0][0] for call in mock_get_dir.call_args_list]
                assert "model" in requested_types
                assert "ensemble" not in requested_types
                # Note: database might be requested by _launch_orchestrator even with empty dbs

    def test_controller_metadata_directory_lazy_creation_pattern(self):
        """Test that metadata directories follow lazy creation pattern"""
        # Create manifest with both model and ensemble
        model = Model("test_model", {}, RunSettings("echo", ["hello"]))
        run_settings = RunSettings("echo", ["world"])
        ensemble = Ensemble("test_ensemble", {}, run_settings=run_settings, replicas=2)
        manifest = Manifest(model, ensemble)

        with (
            patch.object(self.controller, "_jobs") as mock_jobs,
            patch.object(self.controller, "_launch_step") as mock_launch_step,
            patch.object(self.controller, "symlink_output_files") as mock_symlink,
        ):

            mock_jobs.get_db_host_addresses.return_value = {}
            mock_jobs.actively_monitoring = False

            # Track the order of calls to get_entity_metadata_subdirectory
            call_order = []
            original_get_dir = LaunchedManifestBuilder.get_entity_metadata_subdirectory

            def track_calls(self, entity_type):
                call_order.append(entity_type)
                return original_get_dir(self, entity_type)

            with patch.object(
                LaunchedManifestBuilder, "get_entity_metadata_subdirectory", track_calls
            ):
                launched_manifest = self.controller._launch(
                    "test_exp", self.temp_dir, manifest
                )

                # Verify that directories are created in the order they're processed
                # Ensembles are processed before models in the controller
                assert "ensemble" in call_order
                assert "model" in call_order
                # The exact order depends on the controller's processing sequence
