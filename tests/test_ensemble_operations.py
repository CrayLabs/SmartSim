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
from smartsim.builders import Ensemble
from smartsim.builders.utils import strategies


pytestmark = pytest.mark.group_a


@pytest.fixture
def mock_src(test_dir: str):
    """Fixture to create a mock source path."""
    return pathlib.Path(test_dir) / pathlib.Path("mock_src")


# TODO remove when PR 732 is merged
@pytest.fixture
def mock_dest(test_dir: str):
    """Fixture to create a mock destination path."""
    return pathlib.Path(test_dir) / pathlib.Path("mock_dest")


# TODO remove when PR 732 is merged
@pytest.fixture
def copy_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a CopyOperation object."""
    return EnsembleCopyOperation(src=mock_src, dest=mock_dest)


# TODO remove when PR 732 is merged
@pytest.fixture
def symlink_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a SymlinkOperation object."""
    return EnsembleSymlinkOperation(src=mock_src, dest=mock_dest)


# TODO remove when PR 732 is merged
@pytest.fixture
def configure_operation(mock_src: pathlib.Path, mock_dest: pathlib.Path):
    """Fixture to create a Configure object."""
    return EnsembleConfigureOperation(
        src=mock_src, dest=mock_dest, file_parameters={"FOO": ["BAR", "TOE"]}
    )


@pytest.fixture
def ensemble_file_system_operation_set(
    copy_operation: EnsembleCopyOperation,
    symlink_operation: EnsembleSymlinkOperation,
    configure_operation: EnsembleConfigureOperation,
):
    """Fixture to create a FileSysOperationSet object."""
    return EnsembleFileSysOperationSet([copy_operation, symlink_operation, configure_operation])


# @pytest.mark.parametrize(
#     "job_root_path, dest, expected",
#     (
#         pytest.param(
#             pathlib.Path("/valid/root"),
#             pathlib.Path("valid/dest"),
#             "/valid/root/valid/dest",
#             id="Valid paths",
#         ),
#         pytest.param(
#             pathlib.Path("/valid/root"),
#             pathlib.Path(""),
#             "/valid/root",
#             id="Empty destination path",
#         ),
#         pytest.param(
#             pathlib.Path("/valid/root"),
#             None,
#             "/valid/root",
#             id="Empty dest path",
#         ),
#     ),
# )
# def test_create_final_dest_valid(job_root_path, dest, expected):
#     """Test valid path inputs for operations.create_final_dest"""
#     assert create_final_dest(job_root_path, dest) == expected


# @pytest.mark.parametrize(
#     "job_root_path, dest",
#     (
#         pytest.param(None, pathlib.Path("valid/dest"), id="None as root path"),
#         pytest.param(1234, pathlib.Path("valid/dest"), id="Number as root path"),
#         pytest.param(pathlib.Path("valid/dest"), 1234, id="Number as dest"),
#     ),
# )
# def test_create_final_dest_invalid(job_root_path, dest):
#     """Test invalid path inputs for operations.create_final_dest"""
#     with pytest.raises(ValueError):
#         create_final_dest(job_root_path, dest)


# def test_valid_init_generation_context(
#     test_dir: str, generation_context: GenerationContext
# ):
#     """Validate GenerationContext init"""
#     assert isinstance(generation_context, GenerationContext)
#     assert generation_context.job_root_path == pathlib.Path(test_dir)


# def test_invalid_init_generation_context():
#     """Validate GenerationContext init"""
#     with pytest.raises(TypeError):
#         GenerationContext(1234)
#     with pytest.raises(TypeError):
#         GenerationContext("")


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


def test_init_configure_operation(
    mock_src: str, mock_dest: str
):
    """Validate EnsembleConfigureOperation init"""
    ensemble_configure_operation = EnsembleConfigureOperation(mock_src, mock_dest)
    assert isinstance(ensemble_configure_operation, EnsembleConfigureOperation)
    assert ensemble_configure_operation.src == mock_src
    assert ensemble_configure_operation.dest == mock_dest
    assert ensemble_configure_operation.tag == ";"
    assert ensemble_configure_operation.file_parameters == {"FOO": ["BAR", "TOE"]}


def test_init_file_sys_operation_set(file_system_operation_set: EnsembleFileSysOperationSet):
    """Test initialize FileSystemOperationSet"""
    assert isinstance(file_system_operation_set.operations, list)
    assert len(file_system_operation_set.operations) == 3


def test_add_copy_operation(
    file_system_operation_set: EnsembleFileSysOperationSet
):
    """Test FileSystemOperationSet.add_copy"""
    assert len(file_system_operation_set.copy_operations) == 1
    file_system_operation_set.add_copy(src=pathlib.Path("src"))
    assert len(file_system_operation_set.copy_operations) == 2


def test_add_symlink_operation(
    file_system_operation_set: EnsembleFileSysOperationSet
):
    """Test FileSystemOperationSet.add_symlink"""
    assert len(file_system_operation_set.symlink_operations) == 1
    file_system_operation_set.add_symlink(src=pathlib.Path("src"))
    assert len(file_system_operation_set.symlink_operations) == 2


def test_add_configure_operation(
    file_system_operation_set: EnsembleFileSysOperationSet,
):
    """Test FileSystemOperationSet.add_configuration"""
    assert len(file_system_operation_set.configure_operations) == 1
    file_system_operation_set.add_configuration(
        src=pathlib.Path("src"), file_parameters={"FOO": "BAR"}
    )
    assert len(file_system_operation_set.configure_operations) == 2


# @pytest.fixture
# def files(fileutils):
#     path_to_files = fileutils.get_test_conf_path(
#         osp.join("generator_files", "easy", "correct/")
#     )
#     list_of_files_strs = sorted(glob(path_to_files + "/*"))
#     yield [pathlib.Path(str_path) for str_path in list_of_files_strs]


# @pytest.fixture
# def directory(fileutils):
#     directory = fileutils.get_test_conf_path(
#         osp.join("generator_files", "easy", "correct/")
#     )
#     yield [pathlib.Path(directory)]


# @pytest.fixture
# def source(request, files, directory):
#     if request.param == "files":
#         return files
#     elif request.param == "directory":
#         return directory


# @pytest.mark.parametrize(
#     "dest,error",
#     (
#         pytest.param(123, TypeError, id="dest as integer"),
#         pytest.param("", TypeError, id="dest as empty str"),
#         pytest.param("/absolute/path", TypeError, id="dest as absolute str"),
#         pytest.param(
#             pathlib.Path("relative/path"), ValueError, id="dest as relative Path"
#         ),
#         pytest.param(
#             pathlib.Path("/path with spaces"), ValueError, id="dest as Path with spaces"
#         ),
#         # TODO pytest.param(pathlib.Path("/path/with/special!@#"), id="dest as Path with special char"),
#     ),
# )
# @pytest.mark.parametrize("source", ["files", "directory"], indirect=True)
# def test_copy_files_invalid_dest(dest, error, source):
#     """Test invalid copy destination"""
#     with pytest.raises(error):
#         _ = [CopyOperation(src=file, dest=dest) for file in source]


# @pytest.mark.parametrize(
#     "dest,error",
#     (
#         pytest.param(123, TypeError, id="dest as integer"),
#         pytest.param("", TypeError, id="dest as empty str"),
#         pytest.param("/absolute/path", TypeError, id="dest as absolute str"),
#         pytest.param(
#             pathlib.Path("relative/path"), ValueError, id="dest as relative Path"
#         ),
#         pytest.param(
#             pathlib.Path("/path with spaces"), ValueError, id="dest as Path with spaces"
#         ),
#         # TODO pytest.param(pathlib.Path("/path/with/special!@#"), id="dest as Path with special char"),
#     ),
# )
# @pytest.mark.parametrize("source", ["files", "directory"], indirect=True)
# def test_symlink_files_invalid_dest(dest, error, source):
#     """Test invalid symlink destination"""
#     with pytest.raises(error):
#         _ = [SymlinkOperation(src=file, dest=dest) for file in source]


# @pytest.mark.parametrize(
#     "dest,error",
#     (
#         pytest.param(123, TypeError, id="dest as integer"),
#         pytest.param("", TypeError, id="dest as empty str"),
#         pytest.param("/absolute/path", TypeError, id="dest as absolute str"),
#         pytest.param(
#             pathlib.Path("relative/path"), ValueError, id="dest as relative Path"
#         ),
#         pytest.param(
#             pathlib.Path("/path with spaces"), ValueError, id="dest as Path with spaces"
#         ),
#         # TODO pytest.param(pathlib.Path("/path/with/special!@#"), id="dest as Path with special char"),
#     ),
# )
# @pytest.mark.parametrize("source", ["files", "directory"], indirect=True)
# def test_configure_files_invalid_dest(dest, error, source):
#     """Test invalid configure destination"""
#     with pytest.raises(error):
#         _ = [
#             ConfigureOperation(src=file, dest=dest, file_parameters={"FOO": "BAR"})
#             for file in source
#         ]

# bug found that it does not properly permutate if exe_arg_params is not specified
# ISSUE with step, adds one file per application
def test_step_mock():
    ensemble = Ensemble("name", "echo", exe_arg_parameters = {"-N": ["1", "2"]}, permutation_strategy="step")
    ensemble.files.add_configuration(pathlib.Path("src_1"), file_parameters={"FOO":["BAR", "TOE"]})
    ensemble.files.add_configuration(pathlib.Path("src_2"), file_parameters={"CAN":["TOM", "STO"]})
    apps = ensemble._create_applications()
    print(apps)
    for app in apps:
        for config in app.files.configure_operations:
            decoded_dict = base64.b64decode(config.file_parameters)
            print(config.src)
            deserialized_dict = pickle.loads(decoded_dict)
            print(deserialized_dict)

def test_all_perm_mock():
    ensemble = Ensemble("name", "echo", exe_arg_parameters = {"-N": ["1", "2"]}, permutation_strategy="step", replicas=2)
    ensemble.files.add_configuration(pathlib.Path("src_1"), file_parameters={"FOO":["BAR", "TOE"]})
    ensemble.files.add_configuration(pathlib.Path("src_2"), file_parameters={"CAN":["TOM", "STO"]})
    apps = ensemble._create_applications()
    print(len(apps))
    # for app in apps:
    #     for config in app.files.configure_operations:
    #         decoded_dict = base64.b64decode(config.file_parameters)
    #         print(config.src)
    #         deserialized_dict = pickle.loads(decoded_dict)
    #         print(deserialized_dict)

def test_mock():
    ensemble = Ensemble("name", "echo", exe_arg_parameters = {"-N": ["1", "2"]}, permutation_strategy="step")
    file = EnsembleConfigureOperation(src="src", file_parameters={"FOO":["BAR", "TOE"]})
    permutation_strategy = strategies.resolve("all_perm")
    val = ensemble.perm_config_file(file, permutation_strategy)
    print(val)