# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import itertools
import pathlib
import unittest.mock
from glob import glob
from os import path as osp

import pytest

from smartsim._core.commands import Command, CommandList
from smartsim._core.generation.generator import Generator
from smartsim._core.generation.operations.operations import (
    ConfigureOperation,
    CopyOperation,
    FileSysOperationSet,
    GenerationContext,
    SymlinkOperation,
)
from smartsim.entity import SmartSimEntity
from smartsim.launchable import Job

pytestmark = pytest.mark.group_a

ids = set()


_ID_GENERATOR = (str(i) for i in itertools.count())


def random_id():
    return next(_ID_GENERATOR)


@pytest.fixture
def generator_instance(test_dir: str) -> Generator:
    """Instance of Generator"""
    # os.mkdir(root)
    yield Generator(root=pathlib.Path(test_dir))


@pytest.fixture
def mock_index():
    """Fixture to create a mock destination path."""
    return 1


class EchoHelloWorldEntity(SmartSimEntity):
    """A simple smartsim entity that meets the `ExecutableProtocol` protocol"""

    def __init__(self):
        self.name = "entity_name"
        self.files = FileSysOperationSet([])
        self.file_parameters = None

    def as_executable_sequence(self):
        return ("echo", "Hello", "World!")


@pytest.fixture
def mock_job() -> unittest.mock.MagicMock:
    """Fixture to create a mock Job."""
    job = unittest.mock.MagicMock(
        entity=EchoHelloWorldEntity(),
        get_launch_steps=unittest.mock.MagicMock(
            side_effect=lambda: NotImplementedError()
        ),
        spec=Job,
    )
    job.name = "test_job"
    yield job


# UNIT TESTS


def test_init_generator(generator_instance: Generator, test_dir: str):
    """Test Generator init"""
    assert generator_instance.root == pathlib.Path(test_dir)


def test_build_job_base_path(
    generator_instance: Generator, mock_job: unittest.mock.MagicMock, mock_index
):
    """Test Generator._build_job_base_path returns correct path"""
    root_path = generator_instance._build_job_base_path(mock_job, mock_index)
    expected_path = (
        generator_instance.root
        / f"{mock_job.__class__.__name__.lower()}s"
        / f"{mock_job.name}-{mock_index}"
    )
    assert root_path == expected_path


def test_build_job_run_path(
    test_dir: str,
    mock_job: unittest.mock.MagicMock,
    generator_instance: Generator,
    monkeypatch: pytest.MonkeyPatch,
    mock_index,
):
    """Test Generator._build_job_run_path returns correct path"""
    monkeypatch.setattr(
        Generator,
        "_build_job_base_path",
        lambda self, job, job_index: pathlib.Path(test_dir),
    )
    run_path = generator_instance._build_job_run_path(mock_job, mock_index)
    expected_run_path = pathlib.Path(test_dir) / generator_instance.run_directory
    assert run_path == expected_run_path


def test_build_job_log_path(
    test_dir: str,
    mock_job: unittest.mock.MagicMock,
    generator_instance: Generator,
    monkeypatch: pytest.MonkeyPatch,
    mock_index,
):
    """Test Generator._build_job_log_path returns correct path"""
    monkeypatch.setattr(
        Generator,
        "_build_job_base_path",
        lambda self, job, job_index: pathlib.Path(test_dir),
    )
    log_path = generator_instance._build_job_log_path(mock_job, mock_index)
    expected_log_path = pathlib.Path(test_dir) / generator_instance.log_directory
    assert log_path == expected_log_path


def test_build_log_file_path(test_dir: str, generator_instance: Generator):
    """Test Generator._build_log_file_path returns correct path"""
    expected_path = pathlib.Path(test_dir) / "smartsim_params.txt"
    assert generator_instance._build_log_file_path(test_dir) == expected_path


def test_build_out_file_path(
    test_dir: str, generator_instance: Generator, mock_job: unittest.mock.MagicMock
):
    """Test Generator._build_out_file_path returns out path"""
    out_file_path = generator_instance._build_out_file_path(
        pathlib.Path(test_dir), mock_job.name
    )
    assert out_file_path == pathlib.Path(test_dir) / f"{mock_job.name}.out"


def test_build_err_file_path(
    test_dir: str, generator_instance: Generator, mock_job: unittest.mock.MagicMock
):
    """Test Generator._build_err_file_path returns err path"""
    err_file_path = generator_instance._build_err_file_path(
        pathlib.Path(test_dir), mock_job.name
    )
    assert err_file_path == pathlib.Path(test_dir) / f"{mock_job.name}.err"


def test_generate_job(
    mock_job: unittest.mock.MagicMock, generator_instance: Generator, mock_index: int
):
    """Test Generator.generate_job returns correct paths"""
    job_paths = generator_instance.generate_job(mock_job, mock_index)
    assert job_paths.run_path.name == Generator.run_directory
    assert job_paths.out_path.name == f"{mock_job.entity.name}.out"
    assert job_paths.err_path.name == f"{mock_job.entity.name}.err"


def test_execute_commands(generator_instance: Generator):
    """Test Generator._execute_commands subprocess.run"""
    with (
        unittest.mock.patch(
            "smartsim._core.generation.generator.subprocess.run"
        ) as run_process,
    ):
        cmd_list = CommandList(Command(["test", "command"]))
        generator_instance._execute_commands(cmd_list)
        run_process.assert_called_once()


def test_mkdir_file(generator_instance: Generator, test_dir: str):
    """Test Generator._mkdir_file returns correct type and value"""
    cmd = generator_instance._mkdir_file(pathlib.Path(test_dir))
    assert isinstance(cmd, Command)
    assert cmd.command == ["mkdir", "-p", test_dir]


@pytest.mark.parametrize(
    "dest",
    (
        pytest.param(None, id="dest as None"),
        pytest.param(
            pathlib.Path("absolute/path"),
            id="dest as valid path",
        ),
    ),
)
def test_copy_files_valid_dest(
    dest, source, generator_instance: Generator, test_dir: str
):
    to_copy = [CopyOperation(src=file, dest=dest) for file in source]
    gen = GenerationContext(pathlib.Path(test_dir))
    cmd_list = generator_instance._copy_files(files=to_copy, context=gen)
    assert isinstance(cmd_list, CommandList)
    # Extract file paths from commands
    cmd_src_paths = set()
    for cmd in cmd_list.commands:
        src_index = cmd.command.index("copy") + 1
        cmd_src_paths.add(cmd.command[src_index])
    # Assert all file paths are in the command list
    file_paths = {str(file) for file in source}
    assert file_paths == cmd_src_paths, "Not all file paths are in the command list"


@pytest.mark.parametrize(
    "dest",
    (
        pytest.param(None, id="dest as None"),
        pytest.param(
            pathlib.Path("absolute/path"),
            id="dest as valid path",
        ),
    ),
)
def test_symlink_files_valid_dest(
    dest, source, generator_instance: Generator, test_dir: str
):
    to_symlink = [SymlinkOperation(src=file, dest=dest) for file in source]
    gen = GenerationContext(pathlib.Path(test_dir))
    cmd_list = generator_instance._symlink_files(files=to_symlink, context=gen)
    assert isinstance(cmd_list, CommandList)
    # Extract file paths from commands
    cmd_src_paths = set()
    for cmd in cmd_list.commands:
        print(cmd)
        src_index = cmd.command.index("symlink") + 1
        cmd_src_paths.add(cmd.command[src_index])
    # Assert all file paths are in the command list
    file_paths = {str(file) for file in source}
    assert file_paths == cmd_src_paths, "Not all file paths are in the command list"


@pytest.mark.parametrize(
    "dest",
    (
        pytest.param(None, id="dest as None"),
        pytest.param(
            pathlib.Path("absolute/path"),
            id="dest as valid path",
        ),
    ),
)
def test_configure_files_valid_dest(
    dest, source, generator_instance: Generator, test_dir: str
):
    file_param = {
        "5": 10,
        "FIRST": "SECOND",
        "17": 20,
        "65": "70",
        "placeholder": "group leftupper region",
        "1200": "120",
        "VALID": "valid",
    }
    to_configure = [
        ConfigureOperation(src=file, dest=dest, file_parameters=file_param)
        for file in source
    ]
    gen = GenerationContext(pathlib.Path(test_dir))
    cmd_list = generator_instance._configure_files(files=to_configure, context=gen)
    assert isinstance(cmd_list, CommandList)
    # Extract file paths from commands
    cmd_src_paths = set()
    for cmd in cmd_list.commands:
        src_index = cmd.command.index("configure") + 1
        cmd_src_paths.add(cmd.command[src_index])
    # Assert all file paths are in the command list
    file_paths = {str(file) for file in source}
    assert file_paths == cmd_src_paths, "Not all file paths are in the command list"


@pytest.fixture
def run_directory(test_dir, generator_instance):
    return pathlib.Path(test_dir) / generator_instance.run_directory


@pytest.fixture
def log_directory(test_dir, generator_instance):
    return pathlib.Path(test_dir) / generator_instance.log_directory


def test_build_commands(
    generator_instance: Generator,
    run_directory: pathlib.Path,
    log_directory: pathlib.Path,
):
    """Test Generator._build_commands calls internal helper functions"""
    with (
        unittest.mock.patch(
            "smartsim._core.generation.Generator._append_mkdir_commands"
        ) as mock_append_mkdir_commands,
        unittest.mock.patch(
            "smartsim._core.generation.Generator._append_file_operations"
        ) as mock_append_file_operations,
    ):
        generator_instance._build_commands(
            EchoHelloWorldEntity(),
            run_directory,
            log_directory,
        )
        mock_append_mkdir_commands.assert_called_once()
        mock_append_file_operations.assert_called_once()


def test_append_mkdir_commands(
    generator_instance: Generator,
    run_directory: pathlib.Path,
    log_directory: pathlib.Path,
):
    """Test Generator._append_mkdir_commands calls Generator._mkdir_file twice"""
    with (
        unittest.mock.patch(
            "smartsim._core.generation.Generator._mkdir_file"
        ) as mock_mkdir_file,
    ):
        generator_instance._append_mkdir_commands(
            CommandList(),
            run_directory,
            log_directory,
        )
        assert mock_mkdir_file.call_count == 2


def test_append_file_operations(
    context: GenerationContext, generator_instance: Generator
):
    """Test Generator._append_file_operations calls all file operations"""
    with (
        unittest.mock.patch(
            "smartsim._core.generation.Generator._copy_files"
        ) as mock_copy_files,
        unittest.mock.patch(
            "smartsim._core.generation.Generator._symlink_files"
        ) as mock_symlink_files,
        unittest.mock.patch(
            "smartsim._core.generation.Generator._configure_files"
        ) as mock_configure_files,
    ):
        generator_instance._append_file_operations(
            CommandList(),
            EchoHelloWorldEntity(),
            context,
        )
        mock_copy_files.assert_called_once()
        mock_symlink_files.assert_called_once()
        mock_configure_files.assert_called_once()


@pytest.fixture
def paths_to_copy(fileutils):
    paths = fileutils.get_test_conf_path(osp.join("generator_files", "to_copy_dir"))
    yield [pathlib.Path(path) for path in sorted(glob(paths + "/*"))]


@pytest.fixture
def paths_to_symlink(fileutils):
    paths = fileutils.get_test_conf_path(osp.join("generator_files", "to_symlink_dir"))
    yield [pathlib.Path(path) for path in sorted(glob(paths + "/*"))]


@pytest.fixture
def paths_to_configure(fileutils):
    paths = fileutils.get_test_conf_path(
        osp.join("generator_files", "easy", "correct/")
    )
    yield [pathlib.Path(path) for path in sorted(glob(paths + "/*"))]


@pytest.fixture
def context(test_dir: str):
    yield GenerationContext(pathlib.Path(test_dir))


@pytest.fixture
def operations_list(paths_to_copy, paths_to_symlink, paths_to_configure):
    op_list = []
    for file in paths_to_copy:
        op_list.append(CopyOperation(src=file))
    for file in paths_to_symlink:
        op_list.append(SymlinkOperation(src=file))
    for file in paths_to_configure:
        op_list.append(SymlinkOperation(src=file))
    return op_list


@pytest.fixture
def formatted_command_list(operations_list: list, context: GenerationContext):
    new_list = CommandList()
    for file in operations_list:
        new_list.append(file.format(context))
    return new_list


def test_execute_commands(
    operations_list: list, formatted_command_list, generator_instance: Generator
):
    """Test Generator._execute_commands calls with appropriate type and num times"""
    with (
        unittest.mock.patch(
            "smartsim._core.generation.generator.subprocess.run"
        ) as mock_run,
    ):
        generator_instance._execute_commands(formatted_command_list)
        assert mock_run.call_count == len(formatted_command_list)
