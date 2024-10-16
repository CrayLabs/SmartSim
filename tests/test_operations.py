import base64
import os
import pathlib
import pickle

import pytest

from smartsim._core.commands import Command
from smartsim._core.generation.operations.operations import (
    ConfigureOperation,
    CopyOperation,
    FileSysOperationSet,
    GenerationContext,
    SymlinkOperation,
    _check_run_path,
    _create_dest_path,
    configure_cmd,
    copy_cmd,
    default_tag,
    symlink_cmd,
)
from smartsim._core.generation.operations.utils.helpers import check_src_and_dest_path

pytestmark = pytest.mark.group_a


@pytest.fixture
def generation_context(test_dir: str):
    """Fixture to create a GenerationContext object."""
    return GenerationContext(pathlib.Path(test_dir))


@pytest.fixture
def file_system_operation_set(
    copy_operation: CopyOperation,
    symlink_operation: SymlinkOperation,
    configure_operation: ConfigureOperation,
):
    """Fixture to create a FileSysOperationSet object."""
    return FileSysOperationSet([copy_operation, symlink_operation, configure_operation])


# TODO is this test even necessary
@pytest.mark.parametrize(
    "job_run_path, dest",
    (
        pytest.param(
            pathlib.Path("/absolute/src"),
            pathlib.Path("relative/dest"),
            id="Valid paths",
        ),
        pytest.param(
            pathlib.Path("/absolute/src"),
            pathlib.Path(""),
            id="Empty destination path",
        ),
    ),
)
def test_check_src_and_dest_path_valid(job_run_path, dest):
    """Test valid path inputs for helpers.check_src_and_dest_path"""
    check_src_and_dest_path(job_run_path, dest)


@pytest.mark.parametrize(
    "job_run_path, dest, error",
    (
        pytest.param(
            pathlib.Path("relative/src"),
            pathlib.Path("relative/dest"),
            ValueError,
            id="Relative src Path",
        ),
        pytest.param(
            pathlib.Path("/absolute/src"),
            pathlib.Path("/absolute/src"),
            ValueError,
            id="Absolute dest Path",
        ),
        pytest.param(
            123,
            pathlib.Path("relative/dest"),
            TypeError,
            id="non Path src",
        ),
        pytest.param(
            pathlib.Path("/absolute/src"),
            123,
            TypeError,
            id="non Path dest",
        ),
    ),
)
def test_check_src_and_dest_path_invalid(job_run_path, dest, error):
    """Test invalid path inputs for helpers.check_src_and_dest_path"""
    with pytest.raises(error):
        check_src_and_dest_path(job_run_path, dest)


@pytest.mark.parametrize(
    "job_run_path, dest, expected",
    (
        pytest.param(
            pathlib.Path("/absolute/root"),
            pathlib.Path("relative/dest"),
            "/absolute/root/relative/dest",
            id="Valid paths",
        ),
        pytest.param(
            pathlib.Path("/absolute/root"),
            pathlib.Path(""),
            "/absolute/root",
            id="Empty destination path",
        ),
    ),
)
def test_create_dest_path_valid(job_run_path, dest, expected):
    """Test valid path inputs for operations._create_dest_path"""
    assert _create_dest_path(job_run_path, dest) == expected


@pytest.mark.parametrize(
    "job_run_path, error",
    (
        pytest.param(
            pathlib.Path("relative/path"), ValueError, id="Run path is not absolute"
        ),
        pytest.param(1234, TypeError, id="Run path is not pathlib.path"),
    ),
)
def test_check_run_path_invalid(job_run_path, error):
    """Test invalid path inputs for operations._check_run_path"""
    with pytest.raises(error):
        _check_run_path(job_run_path)


def test_valid_init_generation_context(test_dir: str):
    """Validate GenerationContext init"""
    generation_context = GenerationContext(pathlib.Path(test_dir))
    assert isinstance(generation_context, GenerationContext)
    assert generation_context.job_run_path == pathlib.Path(test_dir)


def test_invalid_init_generation_context():
    """Validate GenerationContext init"""
    with pytest.raises(TypeError):
        GenerationContext(1234)
    with pytest.raises(TypeError):
        GenerationContext("")


def test_init_copy_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
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
    test_dir: str,
):
    """Validate CopyOperation.format"""
    exec = copy_operation.format(generation_context)
    assert isinstance(exec, Command)
    assert str(mock_src) in exec.command
    assert copy_cmd in exec.command
    assert _create_dest_path(test_dir, mock_dest) in exec.command


def test_init_symlink_operation(mock_src: str, mock_dest: str):
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
    final_dest = _create_dest_path(generation_context.job_run_path, mock_dest)
    new_dest = os.path.join(final_dest, parent_dir)
    assert new_dest in exec.command


def test_init_configure_operation(mock_src: str, mock_dest: str):
    """Validate ConfigureOperation init"""
    configure_operation = ConfigureOperation(
        src=mock_src, dest=mock_dest, file_parameters={"FOO": "BAR"}
    )
    assert isinstance(configure_operation, ConfigureOperation)
    assert configure_operation.src == mock_src
    assert configure_operation.dest == mock_dest
    assert configure_operation.tag == default_tag
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
    assert _create_dest_path(test_dir, mock_dest) in exec.command


def test_init_file_sys_operation_set(
    copy_operation: CopyOperation,
    symlink_operation: SymlinkOperation,
    configure_operation: ConfigureOperation,
):
    """Test initialize FileSystemOperationSet"""
    file_system_operation_set = FileSysOperationSet(
        [copy_operation, symlink_operation, configure_operation]
    )
    assert isinstance(file_system_operation_set.operations, list)
    assert len(file_system_operation_set.operations) == 3


def test_add_copy_operation(file_system_operation_set: FileSysOperationSet):
    """Test FileSystemOperationSet.add_copy"""
    orig_num_ops = len(file_system_operation_set.copy_operations)
    file_system_operation_set.add_copy(src=pathlib.Path("/src"))
    assert len(file_system_operation_set.copy_operations) == orig_num_ops + 1


def test_add_symlink_operation(file_system_operation_set: FileSysOperationSet):
    """Test FileSystemOperationSet.add_symlink"""
    orig_num_ops = len(file_system_operation_set.symlink_operations)
    file_system_operation_set.add_symlink(src=pathlib.Path("/src"))
    assert len(file_system_operation_set.symlink_operations) == orig_num_ops + 1


def test_add_configure_operation(
    file_system_operation_set: FileSysOperationSet,
):
    """Test FileSystemOperationSet.add_configuration"""
    orig_num_ops = len(file_system_operation_set.configure_operations)
    file_system_operation_set.add_configuration(
        src=pathlib.Path("/src"), file_parameters={"FOO": "BAR"}
    )
    assert len(file_system_operation_set.configure_operations) == orig_num_ops + 1


@pytest.mark.parametrize(
    "dest,error",
    (
        pytest.param(123, TypeError, id="dest as integer"),
        pytest.param("", TypeError, id="dest as empty str"),
        pytest.param(
            pathlib.Path("/absolute/path"), ValueError, id="dest as absolute str"
        ),
    ),
)
def test_copy_files_invalid_dest(dest, error, source):
    """Test invalid copy destination"""
    with pytest.raises(error):
        _ = [CopyOperation(src=file, dest=dest) for file in source]


@pytest.mark.parametrize(
    "src,error",
    (
        pytest.param(123, TypeError, id="src as integer"),
        pytest.param("", TypeError, id="src as empty str"),
        pytest.param(
            pathlib.Path("relative/path"), ValueError, id="src as relative str"
        ),
    ),
)
def test_copy_files_invalid_src(src, error):
    """Test invalid copy source"""
    with pytest.raises(error):
        _ = CopyOperation(src=src)


@pytest.mark.parametrize(
    "dest,error",
    (
        pytest.param(123, TypeError, id="dest as integer"),
        pytest.param("", TypeError, id="dest as empty str"),
        pytest.param(
            pathlib.Path("/absolute/path"), ValueError, id="dest as absolute str"
        ),
    ),
)
def test_symlink_files_invalid_dest(dest, error, source):
    """Test invalid symlink destination"""
    with pytest.raises(error):
        _ = [SymlinkOperation(src=file, dest=dest) for file in source]


@pytest.mark.parametrize(
    "src,error",
    (
        pytest.param(123, TypeError, id="src as integer"),
        pytest.param("", TypeError, id="src as empty str"),
        pytest.param(
            pathlib.Path("relative/path"), ValueError, id="src as relative str"
        ),
    ),
)
def test_symlink_files_invalid_src(src, error):
    """Test invalid symlink source"""
    with pytest.raises(error):
        _ = SymlinkOperation(src=src)


@pytest.mark.parametrize(
    "dest,error",
    (
        pytest.param(123, TypeError, id="dest as integer"),
        pytest.param("", TypeError, id="dest as empty str"),
        pytest.param(
            pathlib.Path("/absolute/path"), ValueError, id="dest as absolute str"
        ),
    ),
)
def test_configure_files_invalid_dest(dest, error, source):
    """Test invalid configure destination"""
    with pytest.raises(error):
        _ = [
            ConfigureOperation(src=file, dest=dest, file_parameters={"FOO": "BAR"})
            for file in source
        ]


@pytest.mark.parametrize(
    "src,error",
    (
        pytest.param(123, TypeError, id="src as integer"),
        pytest.param("", TypeError, id="src as empty str"),
        pytest.param(
            pathlib.Path("relative/path"), ValueError, id="src as relative str"
        ),
    ),
)
def test_configure_files_invalid_src(src, error):
    """Test invalid configure source"""
    with pytest.raises(error):
        _ = ConfigureOperation(src=src, file_parameters={"FOO": "BAR"})
