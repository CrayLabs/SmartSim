from smartsim._core.generation.operations import GenerationContext, CopyOperation, SymlinkOperation, ConfigureOperation, FileSysOperationSet
from smartsim._core.commands import Command
import pathlib
import pytest

# TODO test python protocol?
# TODO add encoded dict into configure op
# TODO create a better way to append the paths together

@pytest.fixture
def generation_context(test_dir: str):
    return GenerationContext(pathlib.Path(test_dir))

@pytest.fixture
def mock_src(test_dir: str):
    return pathlib.Path(test_dir) / pathlib.Path("mock_src")

@pytest.fixture
def mock_dest(test_dir: str):
    return pathlib.Path(test_dir) / pathlib.Path("mock_dest")

@pytest.fixture
def copy_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    return CopyOperation(src=mock_src, dest=mock_dest)

def test_init_generation_context(test_dir: str, generation_context: GenerationContext):
    """Validate GenerationContext init"""
    assert isinstance(generation_context, GenerationContext)
    assert generation_context.job_root_path == pathlib.Path(test_dir)

def test_init_copy_operation(copy_operation: CopyOperation, mock_src: pathlib.Path, mock_dest: pathlib.Path):
    assert copy_operation.src == mock_src
    assert copy_operation.dest == mock_dest

# def test_copy_operation_format(mock_src: str, mock_dest: str, generation_context: GenerationContext):
#     copy_op = CopyOperation(src=mock_src, dest=mock_dest)
#     exec = copy_op.format(generation_context)
#     assert isinstance(exec, Command)
#     # assert (
#     #     str(mock_src)
#     #     and (mock_dest + str(generation_context.job_root_path)) in exec.command
#     # )
    
# def test_init_symlink_operation(mock_src: str, mock_dest: str):
#     symlink_op = SymlinkOperation(src=mock_src, dest=mock_dest)
#     assert symlink_op.src == mock_src
#     assert symlink_op.dest == mock_dest

# def test_symlink_operation_format(mock_src: str, mock_dest: str, generation_context: GenerationContext):
#     symlink_op = SymlinkOperation(src=mock_src, dest=mock_dest)
#     exec = symlink_op.format(generation_context)
#     assert isinstance(exec, Command)
#     # assert (
#     #     str(mock_src)
#     #     and (mock_dest + str(generation_context.job_root_path)) in exec.command
#     # )

# def test_init_configure_operation(mock_src: str, mock_dest: str):
#     config_op = ConfigureOperation(src=mock_src, dest=mock_dest)
#     assert config_op.src == mock_src
#     assert config_op.dest == mock_dest

# def test_configure_operation_format(mock_src: str, mock_dest: str, generation_context: GenerationContext):
#     config_op = ConfigureOperation(src=mock_src, dest=mock_dest)
#     exec = config_op.format(generation_context)
#     assert isinstance(exec, Command)
#     # assert (
#     #     str(mock_src)
#     #     and (mock_dest + str(generation_context.job_root_path)) in exec.command
#     # )
    
# def test_init_file_sys_operation_set():
#     config_op = ConfigureOperation(src=mock_src, dest=mock_dest)
#     file_sys_op_set = FileSysOperationSet([config_op])
#     assert len(file_sys_op_set.operations) == 1

# def test_add_copy_operation():
#     copy_op = CopyOperation(src=mock_src, dest=mock_dest)
#     file_sys_op_set = FileSysOperationSet()
#     file_sys_op_set.add_copy(copy_op)
#     assert len(file_sys_op_set.operations) == 1

# def test_add_symlink_operation():
#     symlink_op = SymlinkOperation(src=mock_src, dest=mock_dest)
#     file_sys_op_set = FileSysOperationSet()
#     file_sys_op_set.add_symlink(symlink_op)
#     assert len(file_sys_op_set.operations) == 1

# def test_add_configure_operation():
#     config_op = ConfigureOperation(src=mock_src, dest=mock_dest)
#     file_sys_op_set = FileSysOperationSet()
#     file_sys_op_set.add_symlink(config_op)
#     assert len(file_sys_op_set.operations) == 1

# def test_copy_property():
#     config_op = ConfigureOperation(src=mock_src, dest=mock_dest)
#     symlink_op = SymlinkOperation(src=mock_src, dest=mock_dest)
#     copy_op = CopyOperation(src=mock_src, dest=mock_dest)
#     file_sys_op_set = FileSysOperationSet([config_op,symlink_op,copy_op])
    
#     assert file_sys_op_set.copy_operations == [copy_op]

# def test_symlink_property():
#     config_op = ConfigureOperation(src=mock_src, dest=mock_dest)
#     symlink_op = SymlinkOperation(src=mock_src, dest=mock_dest)
#     copy_op = CopyOperation(src=mock_src, dest=mock_dest)
#     file_sys_op_set = FileSysOperationSet([config_op,symlink_op,copy_op])
    
#     assert file_sys_op_set.symlink_operations == [symlink_op]

# def test_configure_property():
#     config_op = ConfigureOperation(src=mock_src, dest=mock_dest)
#     symlink_op = SymlinkOperation(src=mock_src, dest=mock_dest)
#     copy_op = CopyOperation(src=mock_src, dest=mock_dest)
#     file_sys_op_set = FileSysOperationSet([config_op,symlink_op,copy_op])
    
#     assert file_sys_op_set.configure_operations == [config_op]