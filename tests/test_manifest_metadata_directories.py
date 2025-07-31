"""Test the metadata directory functionality added to LaunchedManifestBuilder"""

import pathlib
import tempfile
import time
from unittest.mock import patch

import pytest

from smartsim._core.control.manifest import LaunchedManifestBuilder


class TestLaunchedManifestBuilderMetadataDirectories:
    """Test metadata directory properties and methods of LaunchedManifestBuilder"""

    def test_exp_metadata_subdirectory_property(self):
        """Test that exp_metadata_subdirectory returns correct path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lmb = LaunchedManifestBuilder(
                exp_name="test_exp",
                exp_path=temp_dir,
                launcher_name="local",
                run_id="test_run_id",
            )

            expected_path = pathlib.Path(temp_dir) / ".smartsim" / "metadata"
            assert lmb.exp_metadata_subdirectory == expected_path

    def test_run_metadata_subdirectory_property(self):
        """Test that run_metadata_subdirectory returns correct timestamped path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the timestamp to make it predictable
            mock_timestamp = "1234567890123"
            with patch.object(time, "time", return_value=1234567890.123):
                lmb = LaunchedManifestBuilder(
                    exp_name="test_exp",
                    exp_path=temp_dir,
                    launcher_name="local",
                    run_id="test_run_id",
                )

            expected_path = (
                pathlib.Path(temp_dir)
                / ".smartsim"
                / "metadata"
                / f"run_{mock_timestamp}"
            )
            assert lmb.run_metadata_subdirectory == expected_path

    def test_run_metadata_subdirectory_uses_actual_timestamp(self):
        """Test that run_metadata_subdirectory uses actual timestamp from launch"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lmb = LaunchedManifestBuilder(
                exp_name="test_exp",
                exp_path=temp_dir,
                launcher_name="local",
                run_id="test_run_id",
            )

            # Check that the timestamp is reasonable (within last few seconds)
            run_dir_name = lmb.run_metadata_subdirectory.name
            assert run_dir_name.startswith("run_")

            # Extract timestamp and verify it's recent
            timestamp_str = run_dir_name[4:]  # Remove "run_" prefix
            timestamp_ms = int(timestamp_str)
            current_time_ms = int(time.time() * 1000)

            # Should be within 5 seconds of current time
            assert abs(current_time_ms - timestamp_ms) < 5000

    def test_get_entity_metadata_subdirectory_method(self):
        """Test that get_entity_metadata_subdirectory returns correct entity-specific paths"""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_timestamp = "1234567890123"
            with patch.object(time, "time", return_value=1234567890.123):
                lmb = LaunchedManifestBuilder(
                    exp_name="test_exp",
                    exp_path=temp_dir,
                    launcher_name="local",
                    run_id="test_run_id",
                )

            # Test different entity types
            model_dir = lmb.get_entity_metadata_subdirectory("model")
            ensemble_dir = lmb.get_entity_metadata_subdirectory("ensemble")
            database_dir = lmb.get_entity_metadata_subdirectory("database")

            base_path = (
                pathlib.Path(temp_dir)
                / ".smartsim"
                / "metadata"
                / f"run_{mock_timestamp}"
            )

            assert model_dir == base_path / "model"
            assert ensemble_dir == base_path / "ensemble"
            assert database_dir == base_path / "database"

    def test_get_entity_metadata_subdirectory_custom_entity_type(self):
        """Test that get_entity_metadata_subdirectory works with custom entity types"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lmb = LaunchedManifestBuilder(
                exp_name="test_exp",
                exp_path=temp_dir,
                launcher_name="local",
                run_id="test_run_id",
            )

            # Test with custom entity type
            custom_dir = lmb.get_entity_metadata_subdirectory("custom_entity_type")

            expected_path = lmb.run_metadata_subdirectory / "custom_entity_type"
            assert custom_dir == expected_path

    def test_metadata_directory_hierarchy(self):
        """Test that the metadata directory hierarchy is correct"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lmb = LaunchedManifestBuilder(
                exp_name="test_exp",
                exp_path=temp_dir,
                launcher_name="local",
                run_id="test_run_id",
            )

            # Test that the hierarchy is: exp_path/.smartsim/metadata/run_<timestamp>/entity_type
            model_dir = lmb.get_entity_metadata_subdirectory("model")

            # Check path components
            path_parts = model_dir.parts
            assert path_parts[-4] == ".smartsim"
            assert path_parts[-3] == "metadata"
            assert path_parts[-2].startswith("run_")
            assert path_parts[-1] == "model"

    def test_multiple_instances_have_different_timestamps(self):
        """Test that multiple LaunchedManifestBuilder instances have different timestamps"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lmb1 = LaunchedManifestBuilder(
                exp_name="test_exp1",
                exp_path=temp_dir,
                launcher_name="local",
                run_id="test_run_id1",
            )

            # Small delay to ensure different timestamps
            time.sleep(0.001)

            lmb2 = LaunchedManifestBuilder(
                exp_name="test_exp2",
                exp_path=temp_dir,
                launcher_name="local",
                run_id="test_run_id2",
            )

            # Timestamps should be different
            assert lmb1._launch_timestamp != lmb2._launch_timestamp
            assert lmb1.run_metadata_subdirectory != lmb2.run_metadata_subdirectory

    def test_same_instance_consistent_timestamps(self):
        """Test that the same instance always returns consistent timestamps"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lmb = LaunchedManifestBuilder(
                exp_name="test_exp",
                exp_path=temp_dir,
                launcher_name="local",
                run_id="test_run_id",
            )

            # Multiple calls should return the same timestamp
            timestamp1 = lmb._launch_timestamp
            timestamp2 = lmb._launch_timestamp
            assert timestamp1 == timestamp2

            # Multiple calls to run_metadata_subdirectory should be consistent
            run_dir1 = lmb.run_metadata_subdirectory
            run_dir2 = lmb.run_metadata_subdirectory
            assert run_dir1 == run_dir2

    def test_exp_path_with_pathlib(self):
        """Test that metadata directories work correctly when exp_path is a pathlib.Path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_path = pathlib.Path(temp_dir)
            lmb = LaunchedManifestBuilder(
                exp_name="test_exp",
                exp_path=str(exp_path),  # LaunchedManifestBuilder expects string
                launcher_name="local",
                run_id="test_run_id",
            )

            expected_exp_metadata = exp_path / ".smartsim" / "metadata"
            assert lmb.exp_metadata_subdirectory == expected_exp_metadata

    def test_metadata_paths_are_pathlib_paths(self):
        """Test that all metadata directory methods return pathlib.Path objects"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lmb = LaunchedManifestBuilder(
                exp_name="test_exp",
                exp_path=temp_dir,
                launcher_name="local",
                run_id="test_run_id",
            )

            assert isinstance(lmb.exp_metadata_subdirectory, pathlib.Path)
            assert isinstance(lmb.run_metadata_subdirectory, pathlib.Path)
            assert isinstance(
                lmb.get_entity_metadata_subdirectory("model"), pathlib.Path
            )
