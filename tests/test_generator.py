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
import os
import pathlib
import unittest.mock
from glob import glob
from os import listdir
from os import path as osp

import pytest

from smartsim import Experiment
from smartsim._core.commands import Command, CommandList
from smartsim._core.generation.generator import Generator
from smartsim.builders import Ensemble
from smartsim.entity import entity
from smartsim.entity.files import EntityFiles
from smartsim.launchable import Job
from smartsim.settings import LaunchSettings

# TODO Add JobGroup tests when JobGroup becomes a Launchable

pytestmark = pytest.mark.group_a

ids = set()


_ID_GENERATOR = (str(i) for i in itertools.count())


def random_id():
    return next(_ID_GENERATOR)


@pytest.fixture
def get_gen_copy_dir(fileutils):
    yield fileutils.get_test_conf_path(osp.join("generator_files", "to_copy_dir"))


@pytest.fixture
def get_gen_symlink_dir(fileutils):
    yield fileutils.get_test_conf_path(osp.join("generator_files", "to_symlink_dir"))


@pytest.fixture
def get_gen_configure_dir(fileutils):
    yield fileutils.get_test_conf_path(osp.join("generator_files", "tag_dir_template"))


@pytest.fixture
def generator_instance(test_dir: str) -> Generator:
    """Fixture to create an instance of Generator."""
    root = pathlib.Path(test_dir, "temp_id")
    os.mkdir(root)
    yield Generator(root=root)


def get_gen_file(fileutils, filename: str):
    return fileutils.get_test_conf_path(osp.join("generator_files", filename))


class EchoHelloWorldEntity(entity.SmartSimEntity):
    """A simple smartsim entity that meets the `ExecutableProtocol` protocol"""

    def __init__(self):
        self.name = "entity_name"
        self.files = None
        self.file_parameters = None

    def as_executable_sequence(self):
        return ("echo", "Hello", "World!")

    def files():
        return ["file_path"]


@pytest.fixture
def mock_job() -> unittest.mock.MagicMock:
    """Fixture to create a mock Job."""
    job = unittest.mock.MagicMock(
        **{
            "entity": EchoHelloWorldEntity(),
            "name": "test_job",
            "get_launch_steps": unittest.mock.MagicMock(
                side_effect=lambda: NotImplementedError()
            ),
        },
        spec=Job,
    )
    yield job


# UNIT TESTS


def test_init_generator(generator_instance: Generator, test_dir: str):
    """Test Generator init"""
    assert generator_instance.root == pathlib.Path(test_dir) / "temp_id"


def test_build_job_base_path(
    generator_instance: Generator, mock_job: unittest.mock.MagicMock
):
    """Test Generator._build_job_base_path returns correct path"""
    mock_index = 1
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
):
    """Test Generator._build_job_run_path returns correct path"""
    mock_index = 1
    monkeypatch.setattr(
        Generator,
        "_build_job_base_path",
        lambda self, job, job_index: pathlib.Path(test_dir),
    )
    run_path = generator_instance._build_job_run_path(mock_job, mock_index)
    expected_run_path = pathlib.Path(test_dir) / "run"
    assert run_path == expected_run_path


def test_build_job_log_path(
    test_dir: str,
    mock_job: unittest.mock.MagicMock,
    generator_instance: Generator,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test Generator._build_job_log_path returns correct path"""
    mock_index = 1
    monkeypatch.setattr(
        Generator,
        "_build_job_base_path",
        lambda self, job, job_index: pathlib.Path(test_dir),
    )
    log_path = generator_instance._build_job_log_path(mock_job, mock_index)
    expected_log_path = pathlib.Path(test_dir) / "log"
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
    mock_job: unittest.mock.MagicMock,
    generator_instance: Generator,
):
    """Test Generator.generate_job returns correct paths"""
    mock_index = 1
    job_paths = generator_instance.generate_job(mock_job, mock_index)
    assert job_paths.run_path.name == Generator.run_directory
    assert job_paths.out_path.name == f"{mock_job.entity.name}.out"
    assert job_paths.err_path.name == f"{mock_job.entity.name}.err"


def test_build_commands(
    mock_job: unittest.mock.MagicMock, generator_instance: Generator, test_dir: str
):
    """Test Generator._build_commands calls correct helper functions"""
    with (
        unittest.mock.patch(
            "smartsim._core.generation.Generator._copy_files"
        ) as mock_copy_files,
        unittest.mock.patch(
            "smartsim._core.generation.Generator._symlink_files"
        ) as mock_symlink_files,
        unittest.mock.patch(
            "smartsim._core.generation.Generator._write_tagged_files"
        ) as mock_write_tagged_files,
    ):
        generator_instance._build_commands(
            mock_job,
            pathlib.Path(test_dir) / generator_instance.run_directory,
            pathlib.Path(test_dir) / generator_instance.log_directory,
        )
        mock_copy_files.assert_called_once()
        mock_symlink_files.assert_called_once()
        mock_write_tagged_files.assert_called_once()


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


def test_copy_file(generator_instance: Generator, fileutils):
    """Test Generator._copy_files helper function with file"""
    script = fileutils.get_test_conf_path("sleep.py")
    files = EntityFiles(copy=script)
    cmd_list = generator_instance._copy_files(files, generator_instance.root)
    assert isinstance(cmd_list, CommandList)
    assert len(cmd_list) == 1
    assert str(generator_instance.root) and script in cmd_list.commands[0].command


def test_copy_directory(get_gen_copy_dir, generator_instance: Generator):
    """Test Generator._copy_files helper function with directory"""
    files = EntityFiles(copy=get_gen_copy_dir)
    cmd_list = generator_instance._copy_files(files, generator_instance.root)
    assert isinstance(cmd_list, CommandList)
    assert len(cmd_list) == 1
    assert (
        str(generator_instance.root)
        and get_gen_copy_dir in cmd_list.commands[0].command
    )


def test_symlink_file(get_gen_symlink_dir, generator_instance: Generator):
    """Test Generator._symlink_files helper function with file list"""
    symlink_files = sorted(glob(get_gen_symlink_dir + "/*"))
    files = EntityFiles(symlink=symlink_files)
    cmd_list = generator_instance._symlink_files(files, generator_instance.root)
    assert isinstance(cmd_list, CommandList)
    for file, cmd in zip(symlink_files, cmd_list):
        assert file in cmd.command


def test_symlink_directory(generator_instance: Generator, get_gen_symlink_dir):
    """Test Generator._symlink_files helper function with directory"""
    files = EntityFiles(symlink=get_gen_symlink_dir)
    cmd_list = generator_instance._symlink_files(files, generator_instance.root)
    symlinked_folder = generator_instance.root / os.path.basename(get_gen_symlink_dir)
    assert isinstance(cmd_list, CommandList)
    assert str(symlinked_folder) in cmd_list.commands[0].command


def test_write_tagged_file(fileutils, generator_instance: Generator):
    """Test Generator._write_tagged_files helper function with file list"""
    conf_path = fileutils.get_test_conf_path(
        osp.join("generator_files", "easy", "marked/")
    )
    tagged_files = sorted(glob(conf_path + "/*"))
    files = EntityFiles(tagged=tagged_files)
    param_set = {
        "5": 10,
        "FIRST": "SECOND",
        "17": 20,
        "65": "70",
        "placeholder": "group leftupper region",
        "1200": "120",
        "VALID": "valid",
    }
    cmd_list = generator_instance._write_tagged_files(
        files=files, params=param_set, dest=generator_instance.root
    )
    assert isinstance(cmd_list, CommandList)
    for file, cmd in zip(tagged_files, cmd_list):
        assert file in cmd.command


def test_write_tagged_directory(fileutils, generator_instance: Generator):
    """Test Generator._write_tagged_files helper function with directory path"""
    config = get_gen_file(fileutils, "tag_dir_template")
    files = EntityFiles(tagged=[config])
    param_set = {"PARAM0": "param_value_1", "PARAM1": "param_value_2"}
    cmd_list = generator_instance._write_tagged_files(
        files=files, params=param_set, dest=generator_instance.root
    )

    assert isinstance(cmd_list, CommandList)
    assert str(config) in cmd_list.commands[0].command


# INTEGRATED TESTS


def test_exp_private_generate_method(
    mock_job: unittest.mock.MagicMock, test_dir: str, generator_instance: Generator
):
    """Test that Experiment._generate returns expected tuple."""
    mock_index = 1
    exp = Experiment(name="experiment_name", exp_path=test_dir)
    job_paths = exp._generate(generator_instance, mock_job, mock_index)
    assert osp.isdir(job_paths.run_path)
    assert job_paths.out_path.name == f"{mock_job.entity.name}.out"
    assert job_paths.err_path.name == f"{mock_job.entity.name}.err"


def test_generate_ensemble_directory_start(
    test_dir: str, wlmutils, monkeypatch: pytest.MonkeyPatch
):
    """Test that Experiment._generate returns expected tuple."""
    monkeypatch.setattr(
        "smartsim._core.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env, out, err: random_id(),
    )
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.build_jobs(launch_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    exp.start(*job_list)
    run_dir = listdir(test_dir)
    jobs_dir_path = pathlib.Path(test_dir) / run_dir[0] / "jobs"
    list_of_job_dirs = jobs_dir_path.iterdir()
    for job in list_of_job_dirs:
        run_path = jobs_dir_path / job / Generator.run_directory
        assert run_path.is_dir()
        log_path = jobs_dir_path / job / Generator.log_directory
        assert log_path.is_dir()
    ids.clear()


def test_generate_ensemble_copy(
    test_dir: str, wlmutils, monkeypatch: pytest.MonkeyPatch, get_gen_copy_dir
):
    monkeypatch.setattr(
        "smartsim._core.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env, out, err: random_id(),
    )
    ensemble = Ensemble(
        "ensemble-name", "echo", replicas=2, files=EntityFiles(copy=get_gen_copy_dir)
    )
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.build_jobs(launch_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    exp.start(*job_list)
    run_dir = listdir(test_dir)
    jobs_dir = pathlib.Path(test_dir) / run_dir[0] / "jobs"
    job_dir = jobs_dir.iterdir()
    for ensemble_dir in job_dir:
        copy_folder_path = (
            jobs_dir / ensemble_dir / Generator.run_directory / "to_copy_dir"
        )
        assert copy_folder_path.is_dir()
    ids.clear()


def test_generate_ensemble_symlink(
    test_dir: str, wlmutils, monkeypatch: pytest.MonkeyPatch, get_gen_symlink_dir
):
    monkeypatch.setattr(
        "smartsim._core.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env, out, err: random_id(),
    )
    ensemble = Ensemble(
        "ensemble-name",
        "echo",
        replicas=2,
        files=EntityFiles(symlink=get_gen_symlink_dir),
    )
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.build_jobs(launch_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    _ = exp.start(*job_list)
    run_dir = listdir(test_dir)
    jobs_dir = pathlib.Path(test_dir) / run_dir[0] / "jobs"
    job_dir = jobs_dir.iterdir()
    for ensemble_dir in job_dir:
        sym_file_path = pathlib.Path(jobs_dir) / ensemble_dir / "run" / "to_symlink_dir"
        assert sym_file_path.is_dir()
        assert sym_file_path.is_symlink()
        assert os.fspath(sym_file_path.resolve()) == osp.realpath(get_gen_symlink_dir)
    ids.clear()


def test_generate_ensemble_configure(
    test_dir: str, wlmutils, monkeypatch: pytest.MonkeyPatch, get_gen_configure_dir
):
    monkeypatch.setattr(
        "smartsim._core.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env, out, err: random_id(),
    )
    param_set = {"PARAM0": [0, 1], "PARAM1": [2, 3]}
    tagged_files = sorted(glob(get_gen_configure_dir + "/*"))
    ensemble = Ensemble(
        "ensemble-name",
        "echo",
        replicas=1,
        files=EntityFiles(tagged=tagged_files),
        file_parameters=param_set,
    )
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.build_jobs(launch_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    _ = exp.start(*job_list)
    run_dir = listdir(test_dir)
    jobs_dir = pathlib.Path(test_dir) / run_dir[0] / "jobs"

    def _check_generated(param_0, param_1, dir):
        assert dir.is_dir()
        tagged_0 = dir / "tagged_0.sh"
        tagged_1 = dir / "tagged_1.sh"
        assert tagged_0.is_file()
        assert tagged_1.is_file()

        with open(tagged_0) as f:
            line = f.readline()
            assert line.strip() == f'echo "Hello with parameter 0 = {param_0}"'

        with open(tagged_1) as f:
            line = f.readline()
            assert line.strip() == f'echo "Hello with parameter 1 = {param_1}"'

    _check_generated(0, 3, jobs_dir / "ensemble-name-1-1" / Generator.run_directory)
    _check_generated(1, 2, jobs_dir / "ensemble-name-2-2" / Generator.run_directory)
    _check_generated(1, 3, jobs_dir / "ensemble-name-3-3" / Generator.run_directory)
    _check_generated(0, 2, jobs_dir / "ensemble-name-0-0" / Generator.run_directory)
    ids.clear()
