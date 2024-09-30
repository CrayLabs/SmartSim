import pathlib

import pytest

from smartsim._core.commands import Command
from smartsim._core.generation.operations import (
    ConfigureOperation,
    CopyOperation,
    FileSysOperationSet,
    GenerationContext,
    SymlinkOperation,
    create_final_dest,
    copy_cmd,
    symlink_cmd,
    configure_cmd,
)

# QUESTIONS
# TODO test python protocol?
# TODO add encoded dict into configure op
# TODO create a better way to append the paths together
# TODO do I allow the paths to combine if src is empty?


@pytest.fixture
def generation_context(test_dir: str):
    """Fixture to create a GenerationContext object."""
    return GenerationContext(pathlib.Path(test_dir))


@pytest.fixture
def mock_src(test_dir: str):
    """Fixture to create a mock source path."""
    return pathlib.Path(test_dir) / pathlib.Path("mock_src")


@pytest.fixture
def mock_dest(test_dir: str):
    """Fixture to create a mock destination path."""
    return pathlib.Path(test_dir) / pathlib.Path("mock_dest")


@pytest.fixture
def copy_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a CopyOperation object."""
    return CopyOperation(src=mock_src, dest=mock_dest)


@pytest.fixture
def symlink_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a CopyOperation object."""
    return SymlinkOperation(src=mock_src, dest=mock_dest)


@pytest.fixture
def configure_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a CopyOperation object."""
    return ConfigureOperation(src=mock_src, dest=mock_dest, )


@pytest.fixture
def file_system_operation_set(copy_operation: CopyOperation, symlink_operation: SymlinkOperation, configure_operation: ConfigureOperation):
    """Fixture to create a FileSysOperationSet object."""
    return FileSysOperationSet([copy_operation,symlink_operation,configure_operation])


@pytest.mark.parametrize(
    "job_root_path, dest, expected",
    (
        pytest.param(
            pathlib.Path("/valid/root"),
            pathlib.Path("valid/dest"),
            "/valid/root/valid/dest",
            id="Valid paths",
        ),
        # pytest.param(
        #     pathlib.Path("/valid/root/"),
        #     pathlib.Path("/valid/dest.txt"),
        #     "/valid/root/valid/dest.txt",
        #     id="Valid_file_path",
        # ),
        pytest.param(
            pathlib.Path("/valid/root"),
            pathlib.Path(""),
            "/valid/root",
            id="Empty destination path",
        ),
        pytest.param(
            pathlib.Path("/valid/root"),
            None,
            "/valid/root",
            id="Empty dest path",
        ),
    ),
)
def test_create_final_dest_valid(job_root_path, dest, expected):
    """Test valid path inputs for operations.create_final_dest"""
    assert create_final_dest(job_root_path, dest) == expected


@pytest.mark.parametrize(
    "job_root_path, dest",
    (
        pytest.param(None, pathlib.Path("valid/dest"), id="None as root path"),
        pytest.param("", pathlib.Path("valid/dest"), id="Empty str as root path"),
        pytest.param(pathlib.Path("/invalid/root.py"), pathlib.Path("valid/dest"), id="File as root path"),
    ),
)
def test_create_final_dest_invalid(job_root_path, dest):
    """Test invalid path inputs for operations.create_final_dest"""
    with pytest.raises(ValueError):
        create_final_dest(job_root_path, dest)


def test_init_generation_context(test_dir: str, generation_context: GenerationContext):
    """Validate GenerationContext init"""
    assert isinstance(generation_context, GenerationContext)
    assert generation_context.job_root_path == pathlib.Path(test_dir)


def test_init_copy_operation(
    copy_operation: CopyOperation, mock_src: pathlib.Path, mock_dest: pathlib.Path
):
    """Validate CopyOperation init"""
    assert isinstance(copy_operation, CopyOperation)
    assert copy_operation.src == mock_src
    assert copy_operation.dest == mock_dest


def test_copy_operation_format(copy_operation: CopyOperation, mock_src: str, mock_dest: str, generation_context: GenerationContext):
    """Validate CopyOperation.format"""
    exec = copy_operation.format(generation_context)
    assert isinstance(exec, Command)
    assert str(mock_src) in exec.command
    assert copy_cmd in exec.command
    assert create_final_dest(mock_src, mock_dest) in exec.command

def test_init_symlink_operation(symlink_operation: SymlinkOperation, mock_src: str, mock_dest: str):
    """Validate SymlinkOperation init"""
    assert isinstance(symlink_operation, SymlinkOperation)
    assert symlink_operation.src == mock_src
    assert symlink_operation.dest == mock_dest

def test_symlink_operation_format(symlink_operation: SymlinkOperation, mock_src: str, mock_dest: str, generation_context: GenerationContext):
    """Validate SymlinkOperation.format"""
    exec = symlink_operation.format(generation_context)
    assert isinstance(exec, Command)
    assert str(mock_src) in exec.command
    assert symlink_cmd in exec.command
    assert create_final_dest(mock_src, mock_dest) in exec.command

def test_init_configure_operation(configure_operation: ConfigureOperation, mock_src: str, mock_dest: str):
    """Validate ConfigureOperation init"""
    assert isinstance(configure_operation, ConfigureOperation)
    assert configure_operation.src == mock_src
    assert configure_operation.dest == mock_dest
    assert configure_operation.tag == ";"

def test_configure_operation_format(configure_operation: ConfigureOperation, mock_src: str, mock_dest: str, generation_context: GenerationContext):
    """Validate ConfigureOperation.format"""
    exec = configure_operation.format(generation_context)
    assert isinstance(exec, Command)
    assert str(mock_src) in exec.command
    assert configure_cmd in exec.command
    assert create_final_dest(mock_src, mock_dest) in exec.command

def test_init_file_sys_operation_set(file_system_operation_set: FileSysOperationSet):
    assert isinstance(file_system_operation_set.operations, list)
    assert len(file_system_operation_set.operations) == 3

def test_add_copy_operation(file_system_operation_set: FileSysOperationSet, copy_operation: CopyOperation):
    assert len(file_system_operation_set.copy_operations) == 1
    file_system_operation_set.add_copy(copy_operation)
    assert len(file_system_operation_set.copy_operations) == 2

def test_add_symlink_operation(file_system_operation_set: FileSysOperationSet, symlink_operation: SymlinkOperation):
    assert len(file_system_operation_set.symlink_operations) == 1
    file_system_operation_set.add_symlink(symlink_operation)
    assert len(file_system_operation_set.symlink_operations) == 2

def test_add_configure_operation(file_system_operation_set: FileSysOperationSet, configure_operation: ConfigureOperation):
    assert len(file_system_operation_set.configure_operations) == 1
    file_system_operation_set.add_configuration(configure_operation)
    assert len(file_system_operation_set.configure_operations) == 2
