import base64
import os
import pathlib
import pickle
from glob import glob
from os import path as osp
import pickle

import pytest

from smartsim._core.commands import Command
from smartsim._core.generation.operations.ensemble_operations import (
    EnsembleCopyOperation,
    EnsembleSymlinkOperation,
    EnsembleConfigureOperation,
    EnsembleFileSysOperationSet
)
from smartsim._core.generation.operations.operations import default_tag
from smartsim.builders import Ensemble
from smartsim.builders.utils import strategies


pytestmark = pytest.mark.group_a

# TODO missing test for _filter

@pytest.fixture
def ensemble_copy_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a EnsembleCopyOperation object."""
    return EnsembleCopyOperation(src=mock_src, dest=mock_dest)


@pytest.fixture
def ensemble_symlink_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a EnsembleSymlinkOperation object."""
    return EnsembleSymlinkOperation(src=mock_src, dest=mock_dest)


@pytest.fixture
def ensemble_configure_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a EnsembleConfigureOperation object."""
    return EnsembleConfigureOperation(
        src=mock_src, dest=mock_dest, file_parameters={"FOO": ["BAR", "TOE"]}
    )


@pytest.fixture
def ensemble_file_system_operation_set(
    ensemble_copy_operation: EnsembleCopyOperation,
    ensemble_symlink_operation: EnsembleSymlinkOperation,
    ensemble_configure_operation: EnsembleConfigureOperation,
):
    """Fixture to create a FileSysOperationSet object."""
    return EnsembleFileSysOperationSet([ensemble_copy_operation, ensemble_symlink_operation, ensemble_configure_operation])


def test_init_ensemble_copy_operation(
    mock_src: pathlib.Path, mock_dest: pathlib.Path
):
    """Validate EnsembleCopyOperation init"""
    ensemble_copy_operation = EnsembleCopyOperation(mock_src, mock_dest)
    assert isinstance(ensemble_copy_operation, EnsembleCopyOperation)
    assert ensemble_copy_operation.src == mock_src
    assert ensemble_copy_operation.dest == mock_dest


def test_init_ensemble_symlink_operation(
    mock_src: str, mock_dest: str
):
    """Validate EnsembleSymlinkOperation init"""
    ensemble_symlink_operation = EnsembleSymlinkOperation(mock_src, mock_dest)
    assert isinstance(ensemble_symlink_operation, EnsembleSymlinkOperation)
    assert ensemble_symlink_operation.src == mock_src
    assert ensemble_symlink_operation.dest == mock_dest


def test_init_ensemble_configure_operation(
    mock_src: str
):
    """Validate EnsembleConfigureOperation init"""
    ensemble_configure_operation = EnsembleConfigureOperation(mock_src, file_parameters={"FOO": ["BAR", "TOE"]})
    assert isinstance(ensemble_configure_operation, EnsembleConfigureOperation)
    assert ensemble_configure_operation.src == mock_src
    assert ensemble_configure_operation.dest == None
    assert ensemble_configure_operation.tag == default_tag
    assert ensemble_configure_operation.file_parameters == {"FOO": ["BAR", "TOE"]}


def test_init_ensemble_file_sys_operation_set(
    copy_operation: EnsembleCopyOperation,
    symlink_operation: EnsembleSymlinkOperation,
    configure_operation: EnsembleConfigureOperation):
    """Test initialize EnsembleFileSysOperationSet"""
    ensemble_fs_op_set = EnsembleFileSysOperationSet([copy_operation, symlink_operation, configure_operation])
    assert isinstance(ensemble_fs_op_set.operations, list)
    assert len(ensemble_fs_op_set.operations) == 3


def test_add_ensemble_copy_operation(ensemble_file_system_operation_set: EnsembleFileSysOperationSet):
    """Test EnsembleFileSysOperationSet.add_copy"""
    orig_num_ops = len(ensemble_file_system_operation_set.copy_operations)
    ensemble_file_system_operation_set.add_copy(src=pathlib.Path("/src"))
    assert len(ensemble_file_system_operation_set.copy_operations) == orig_num_ops + 1


def test_add_ensemble_symlink_operation(ensemble_file_system_operation_set: EnsembleFileSysOperationSet):
    """Test EnsembleFileSysOperationSet.add_symlink"""
    orig_num_ops = len(ensemble_file_system_operation_set.symlink_operations)
    ensemble_file_system_operation_set.add_symlink(src=pathlib.Path("/src"))
    assert len(ensemble_file_system_operation_set.symlink_operations) == orig_num_ops + 1


def test_add_ensemble_configure_operation(
    ensemble_file_system_operation_set: EnsembleFileSysOperationSet,
):
    """Test FileSystemOperationSet.add_configuration"""
    orig_num_ops = len(ensemble_file_system_operation_set.configure_operations)
    ensemble_file_system_operation_set.add_configuration(src=pathlib.Path("/src"), file_parameters={"FOO": "BAR"})
    assert len(ensemble_file_system_operation_set.configure_operations) == orig_num_ops + 1


@pytest.mark.parametrize(
    "dest,error",
    (
        pytest.param(123, TypeError, id="dest as integer"),
        pytest.param("", TypeError, id="dest as empty str"),
        pytest.param(pathlib.Path("/absolute/path"), ValueError, id="dest as absolute str"),
    ),
)
def test_ensemble_copy_files_invalid_dest(dest, error, source):
    """Test invalid copy destination"""
    with pytest.raises(error):
        _ = [EnsembleCopyOperation(src=file, dest=dest) for file in source]


@pytest.mark.parametrize(
    "src,error",
    (
        pytest.param(123, TypeError, id="src as integer"),
        pytest.param("", TypeError, id="src as empty str"),
        pytest.param(pathlib.Path("relative/path"), ValueError, id="src as relative str"),
    ),
)
def test_ensemble_copy_files_invalid_src(src, error):
    """Test invalid copy source"""
    with pytest.raises(error):
        _ = EnsembleCopyOperation(src=src)


@pytest.mark.parametrize(
    "dest,error",
    (
        pytest.param(123, TypeError, id="dest as integer"),
        pytest.param("", TypeError, id="dest as empty str"),
        pytest.param(pathlib.Path("/absolute/path"), ValueError, id="dest as absolute str"),
    ),
)
def test_ensemble_symlink_files_invalid_dest(dest, error, source):
    """Test invalid symlink destination"""
    with pytest.raises(error):
        _ = [EnsembleSymlinkOperation(src=file, dest=dest) for file in source]


@pytest.mark.parametrize(
    "src,error",
    (
        pytest.param(123, TypeError, id="src as integer"),
        pytest.param("", TypeError, id="src as empty str"),
        pytest.param(pathlib.Path("relative/path"), ValueError, id="src as relative str"),
    ),
)
def test_ensemble_symlink_files_invalid_src(src, error):
    """Test invalid symlink source"""
    with pytest.raises(error):
        _ = EnsembleSymlinkOperation(src=src)


@pytest.mark.parametrize(
    "dest,error",
    (
        pytest.param(123, TypeError, id="dest as integer"),
        pytest.param("", TypeError, id="dest as empty str"),
        pytest.param(pathlib.Path("/absolute/path"), ValueError, id="dest as absolute str"),
    ),
)
def test_ensemble_configure_files_invalid_dest(dest, error, source):
    """Test invalid configure destination"""
    with pytest.raises(error):
        _ = [
            EnsembleConfigureOperation(src=file, dest=dest, file_parameters={"FOO": "BAR"})
            for file in source
        ]


@pytest.mark.parametrize(
    "src,error",
    (
        pytest.param(123, TypeError, id="src as integer"),
        pytest.param("", TypeError, id="src as empty str"),
        pytest.param(pathlib.Path("relative/path"), ValueError, id="src as relative str"),
    ),
)
def test_ensemble_configure_files_invalid_src(src, error):
    """Test invalid configure source"""
    with pytest.raises(error):
        _ = EnsembleConfigureOperation(src=src, file_parameters={"FOO":["BAR", "TOE"]})
