import base64
import os
import pathlib
import pickle
from glob import glob
from os import path as osp

import pytest

from smartsim._core.commands import Command
from smartsim._core.generation.operations import (
    ConfigureOperation,
    CopyOperation,
    FileSysOperationSet,
    GenerationContext,
    SymlinkOperation,
    configure_cmd,
    copy_cmd,
    _create_final_dest,
    symlink_cmd,
)

# TODO ADD CHECK TO ENFORCE SRC AS RELATIVE

pytestmark = pytest.mark.group_a


@pytest.fixture
def generation_context(test_dir: str):
    """Fixture to create a GenerationContext object."""
    return GenerationContext(pathlib.Path(test_dir))


@pytest.fixture
def mock_src(test_dir: str):
    """Fixture to create a mock source path."""
    return pathlib.Path(test_dir) / pathlib.Path("mock_src")


@pytest.fixture
def mock_dest():
    """Fixture to create a mock destination path."""
    return pathlib.Path("mock_dest")


@pytest.fixture
def copy_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a CopyOperation object."""
    return CopyOperation(src=mock_src, dest=mock_dest)


@pytest.fixture
def symlink_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a SymlinkOperation object."""
    return SymlinkOperation(src=mock_src, dest=mock_dest)


@pytest.fixture
def configure_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a Configure object."""
    return ConfigureOperation(
        src=mock_src, dest=mock_dest, file_parameters={"FOO": "BAR"}
    )


@pytest.fixture
def file_system_operation_set(
    copy_operation: CopyOperation,
    symlink_operation: SymlinkOperation,
    configure_operation: ConfigureOperation,
):
    """Fixture to create a FileSysOperationSet object."""
    return FileSysOperationSet([copy_operation, symlink_operation, configure_operation])


@pytest.mark.parametrize(
    "job_root_path, dest, expected",
    (
        pytest.param(
            pathlib.Path("/valid/root"),
            pathlib.Path("valid/dest"),
            "/valid/root/valid/dest",
            id="Valid paths",
        ),
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
    """Test valid path inputs for operations._create_final_dest"""
    assert _create_final_dest(job_root_path, dest) == expected


@pytest.mark.parametrize(
    "job_root_path, dest",
    (
        pytest.param(None, pathlib.Path("valid/dest"), id="None as root path"),
        pytest.param(1234, pathlib.Path("valid/dest"), id="Number as root path"),
        pytest.param(pathlib.Path("valid/dest"), 1234, id="Number as dest"),
    ),
)
def test_create_final_dest_invalid(job_root_path, dest):
    """Test invalid path inputs for operations._create_final_dest"""
    with pytest.raises(TypeError):
        _create_final_dest(job_root_path, dest)


def test_valid_init_generation_context(
    test_dir: str
):
    """Validate GenerationContext init"""
    generation_context = GenerationContext(pathlib.Path(test_dir))
    assert isinstance(generation_context, GenerationContext)
    assert generation_context.job_root_path == pathlib.Path(test_dir)


def test_invalid_init_generation_context():
    """Validate GenerationContext init"""
    with pytest.raises(TypeError):
        GenerationContext(1234)
    with pytest.raises(TypeError):
        GenerationContext("")


def test_init_copy_operation(
    mock_src: pathlib.Path, mock_dest: pathlib.Path
):
    """Validate CopyOperation init"""
    copy_operation = CopyOperation(mock_src, mock_dest)
    assert isinstance(copy_operation, CopyOperation)
    assert copy_operation.src == mock_src
    assert copy_operation.dest == mock_dest


def test_copy_operation_format(
    copy_operation: CopyOperation,
    mock_dest: str,
    mock_src: str,
    generation_context: GenerationContext,
    test_dir: str
):
    """Validate CopyOperation.format"""
    exec = copy_operation.format(generation_context)
    assert isinstance(exec, Command)
    assert str(mock_src) in exec.command
    assert copy_cmd in exec.command
    assert _create_final_dest(test_dir, mock_dest) in exec.command


def test_init_symlink_operation(
    mock_src: str, mock_dest: str
):
    """Validate SymlinkOperation init"""
    symlink_operation = SymlinkOperation(mock_src, mock_dest)
    assert isinstance(symlink_operation, SymlinkOperation)
    assert symlink_operation.src == mock_src
    assert symlink_operation.dest == mock_dest


def test_symlink_operation_format(
    symlink_operation: SymlinkOperation,
    mock_src: str,
    mock_dest: str,
    generation_context: GenerationContext,
):
    """Validate SymlinkOperation.format"""
    exec = symlink_operation.format(generation_context)
    assert isinstance(exec, Command)
    assert str(mock_src) in exec.command
    assert symlink_cmd in exec.command

    normalized_path = os.path.normpath(mock_src)
    parent_dir = os.path.dirname(normalized_path)
    final_dest = _create_final_dest(generation_context.job_root_path, mock_dest)
    new_dest = os.path.join(final_dest, parent_dir)
    assert new_dest in exec.command


def test_init_configure_operation(
    mock_src: str, mock_dest: str
):
    """Validate ConfigureOperation init"""
    configure_operation = ConfigureOperation(src=mock_src, dest=mock_dest,file_parameters={"FOO": "BAR"})
    assert isinstance(configure_operation, ConfigureOperation)
    assert configure_operation.src == mock_src
    assert configure_operation.dest == mock_dest
    assert configure_operation.tag == ";"
    decoded_dict = base64.b64decode(configure_operation.file_parameters.encode("ascii"))
    unpickled_dict = pickle.loads(decoded_dict)
    assert unpickled_dict == {"FOO": "BAR"}


def test_configure_operation_format(
    configure_operation: ConfigureOperation,
    test_dir: str,
    mock_dest: str,
    mock_src: str,
    generation_context: GenerationContext,
):
    """Validate ConfigureOperation.format"""
    exec = configure_operation.format(generation_context)
    assert isinstance(exec, Command)
    assert str(mock_src) in exec.command
    assert configure_cmd in exec.command
    assert _create_final_dest(test_dir, mock_dest) in exec.command


def test_init_file_sys_operation_set(
    copy_operation: CopyOperation,
    symlink_operation: SymlinkOperation,
    configure_operation: ConfigureOperation):
    """Test initialize FileSystemOperationSet"""
    file_system_operation_set = FileSysOperationSet([copy_operation,symlink_operation,configure_operation])
    assert isinstance(file_system_operation_set.operations, list)
    assert len(file_system_operation_set.operations) == 3


def test_add_copy_operation(
    file_system_operation_set: FileSysOperationSet, copy_operation: CopyOperation
):
    """Test FileSystemOperationSet.add_copy"""
    assert len(file_system_operation_set.copy_operations) == 1
    file_system_operation_set.add_copy(src=pathlib.Path("/src"))
    assert len(file_system_operation_set.copy_operations) == 2


def test_add_symlink_operation(
    file_system_operation_set: FileSysOperationSet, symlink_operation: SymlinkOperation
):
    """Test FileSystemOperationSet.add_symlink"""
    assert len(file_system_operation_set.symlink_operations) == 1
    file_system_operation_set.add_symlink(src=pathlib.Path("src"))
    assert len(file_system_operation_set.symlink_operations) == 2


def test_add_configure_operation(
    file_system_operation_set: FileSysOperationSet,
    configure_operation: ConfigureOperation,
):
    """Test FileSystemOperationSet.add_configuration"""
    assert len(file_system_operation_set.configure_operations) == 1
    file_system_operation_set.add_configuration(
        src=pathlib.Path("src"), file_parameters={"FOO": "BAR"}
    )
    assert len(file_system_operation_set.configure_operations) == 2


# might change this to files that can be configured
@pytest.fixture
def files(fileutils):
    path_to_files = fileutils.get_test_conf_path(
        osp.join("generator_files", "easy", "correct/")
    )
    list_of_files_strs = sorted(glob(path_to_files + "/*"))
    yield [pathlib.Path(str_path) for str_path in list_of_files_strs]


@pytest.fixture
def directory(fileutils):
    directory = fileutils.get_test_conf_path(
        osp.join("generator_files", "easy", "correct/")
    )
    yield [pathlib.Path(directory)]


@pytest.fixture
def source(request, files, directory):
    if request.param == "files":
        return files
    elif request.param == "directory":
        return directory


@pytest.mark.parametrize(
    "dest,error",
    (
        pytest.param(123, TypeError, id="dest as integer"),
        pytest.param("", TypeError, id="dest as empty str"),
        pytest.param("/absolute/path", TypeError, id="dest as absolute str"),
    ),
)
@pytest.mark.parametrize("source", ["files", "directory"], indirect=True)
def test_copy_files_invalid_dest(dest, error, source):
    """Test invalid copy destination"""
    with pytest.raises(error):
        _ = [CopyOperation(src=file, dest=dest) for file in source]


@pytest.mark.parametrize(
    "dest,error",
    (
        pytest.param(123, TypeError, id="dest as integer"),
        pytest.param("", TypeError, id="dest as empty str"),
        pytest.param("/absolute/path", TypeError, id="dest as absolute str"),
    ),
)
@pytest.mark.parametrize("source", ["files", "directory"], indirect=True)
def test_symlink_files_invalid_dest(dest, error, source):
    """Test invalid symlink destination"""
    with pytest.raises(error):
        _ = [SymlinkOperation(src=file, dest=dest) for file in source]


@pytest.mark.parametrize(
    "dest,error",
    (
        pytest.param(123, TypeError, id="dest as integer"),
        pytest.param("", TypeError, id="dest as empty str"),
        pytest.param("/absolute/path", TypeError, id="dest as absolute str"),
    ),
)
@pytest.mark.parametrize("source", ["files", "directory"], indirect=True)
def test_configure_files_invalid_dest(dest, error, source):
    """Test invalid configure destination"""
    with pytest.raises(error):
        _ = [
            ConfigureOperation(src=file, dest=dest, file_parameters={"FOO": "BAR"})
            for file in source
        ]
