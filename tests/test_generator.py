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

import filecmp
import itertools
import os
import pathlib
import random
from glob import glob
from os import listdir
from os import path as osp

import pytest

from smartsim import Experiment
from smartsim._core.generation.generator import Generator
from smartsim.entity import Application, Ensemble, entity
from smartsim.entity.files import EntityFiles
from smartsim.launchable import Job
from smartsim.launchable.basejob import BaseJob
from smartsim.settings import LaunchSettings

# TODO Add JobGroup tests when JobGroup becomes a Launchable

pytestmark = pytest.mark.group_a

ids = set()

def random_id():
    while True:
        num = str(random.randint(1, 100))
        if num not in ids:
            ids.add(num)
            return num


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
def generator_instance(test_dir) -> Generator:
    """Fixture to create an instance of Generator."""
    root = pathlib.Path(test_dir, "temp_id")
    os.mkdir(root)
    yield Generator(root=root)


def get_gen_file(fileutils, filename):
    return fileutils.get_test_conf_path(osp.join("generator_files", filename))


class EchoHelloWorldEntity(entity.SmartSimEntity):
    """A simple smartsim entity that meets the `ExecutableProtocol` protocol"""

    def __init__(self):
        self.name = "entity_name"


class MockJob(BaseJob):
    """Mock Job for testing."""

    def __init__(self):
        self.name = "test_job"
        self.index = 1
        self.entity = EchoHelloWorldEntity()

    def get_launch_steps(self):
        raise NotImplementedError


# UNIT TESTS


def test_init_generator(generator_instance, test_dir):
    """Test Generator init"""
    assert generator_instance.root == pathlib.Path(test_dir) / "temp_id"


def test_generate_job_root(generator_instance):
    """Test Generator._generate_job_root returns correct path"""
    mock_job = MockJob()
    root_path = generator_instance._generate_job_root(mock_job, mock_job.index)
    expected_path = (
        generator_instance.root
        / f"{MockJob.__name__.lower()}s"
        / f"{mock_job.name}-{mock_job.index}"
    )
    assert root_path == expected_path


def test_generate_run_path(generator_instance):
    """Test Generator._generate_run_path returns correct path"""
    mock_job = MockJob()
    run_path = generator_instance._generate_run_path(mock_job, mock_job.index)
    expected_path = (
        generator_instance.root
        / f"{MockJob.__name__.lower()}s"
        / f"{mock_job.name}-{mock_job.index}"
        / "run"
    )
    assert run_path == expected_path


def test_generate_log_path(generator_instance):
    """Test Generator._generate_log_path returns correct path"""
    mock_job = MockJob()
    log_path = generator_instance._generate_log_path(mock_job, mock_job.index)
    expected_path = (
        generator_instance.root
        / f"{MockJob.__name__.lower()}s"
        / f"{mock_job.name}-{mock_job.index}"
        / "log"
    )
    assert log_path == expected_path


def test_log_file_path(generator_instance):
    """Test Generator._log_file returns correct path"""
    base_path = "/tmp"
    expected_path = osp.join(base_path, "smartsim_params.txt")
    assert generator_instance._log_file(base_path) == pathlib.Path(expected_path)


def test_output_files(generator_instance):
    """Test Generator._output_files returns err/out paths"""
    base_path = pathlib.Path("/tmp")
    out_file_path, err_file_path = generator_instance._output_files(
        base_path, MockJob().name
    )
    assert out_file_path == base_path / f"{MockJob().name}.out"
    assert err_file_path == base_path / f"{MockJob().name}.err"


def test_copy_file(generator_instance, fileutils):
    """Test Generator._copy_files helper function with file list"""
    script = fileutils.get_test_conf_path("sleep.py")
    files = EntityFiles(copy=script)
    generator_instance._copy_files(files, generator_instance.root)
    expected_file = generator_instance.root / "sleep.py"
    assert osp.isfile(expected_file)


def test_copy_directory(get_gen_copy_dir, generator_instance):
    """Test Generator._copy_files helper function with directory"""
    files = EntityFiles(copy=get_gen_copy_dir)
    generator_instance._copy_files(files, generator_instance.root)
    copied_folder = generator_instance.root / os.path.basename(get_gen_copy_dir)
    assert osp.isdir(copied_folder)


def test_symlink_directory(generator_instance, get_gen_symlink_dir):
    """Test Generator._symlink_files helper function with directory"""
    files = EntityFiles(symlink=get_gen_symlink_dir)
    generator_instance._symlink_files(files, generator_instance.root)
    symlinked_folder = generator_instance.root / os.path.basename(get_gen_symlink_dir)
    assert osp.isdir(symlinked_folder)
    assert symlinked_folder.is_symlink()
    assert os.fspath(symlinked_folder.resolve()) == osp.realpath(get_gen_symlink_dir)
    for written, correct in itertools.zip_longest(
        listdir(get_gen_symlink_dir), listdir(symlinked_folder)
    ):
        assert written == correct


def test_symlink_file(get_gen_symlink_dir, generator_instance):
    """Test Generator._symlink_files helper function with file list"""
    symlink_files = sorted(glob(get_gen_symlink_dir + "/*"))
    files = EntityFiles(symlink=symlink_files)
    generator_instance._symlink_files(files, generator_instance.root)
    symlinked_file = generator_instance.root / os.path.basename(symlink_files[0])
    assert osp.isfile(symlinked_file)
    assert symlinked_file.is_symlink()
    assert os.fspath(symlinked_file.resolve()) == osp.join(
        osp.realpath(get_gen_symlink_dir), "mock2.txt"
    )


def test_write_tagged_file(fileutils, generator_instance):
    """Test Generator._write_tagged_files helper function with file list"""
    conf_path = fileutils.get_test_conf_path(
        osp.join("generator_files", "easy", "marked/")
    )
    tagged_files = sorted(glob(conf_path + "/*"))
    correct_path = fileutils.get_test_conf_path(
        osp.join("generator_files", "easy", "correct/")
    )
    correct_files = sorted(glob(correct_path + "/*"))
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
    generator_instance._write_tagged_files(
        files=files, params=param_set, dest=generator_instance.root
    )
    configured_files = sorted(glob(os.path.join(generator_instance.root) + "/*"))
    for written, correct in itertools.zip_longest(configured_files, correct_files):
        assert filecmp.cmp(written, correct)


def test_write_tagged_directory(fileutils, generator_instance):
    """Test Generator._write_tagged_files helper function with directory path"""
    config = get_gen_file(fileutils, "tag_dir_template")
    files = EntityFiles(tagged=config)
    param_set = {"PARAM0": "param_value_1", "PARAM1": "param_value_2"}
    generator_instance._write_tagged_files(
        files=files, params=param_set, dest=generator_instance.root
    )

    def _check_generated(param_0, param_1):
        assert osp.isdir(osp.join(generator_instance.root, "nested_0"))
        assert osp.isdir(osp.join(generator_instance.root, "nested_1"))

        with open(osp.join(generator_instance.root, "nested_0", "tagged_0.sh")) as f:
            line = f.readline()
            assert line.strip() == f'echo "Hello with parameter 0 = {param_0}"'

        with open(osp.join(generator_instance.root, "nested_1", "tagged_1.sh")) as f:
            line = f.readline()
            assert line.strip() == f'echo "Hello with parameter 1 = {param_1}"'

    _check_generated("param_value_1", "param_value_2")


def test_generate_job(generator_instance, monkeypatch: pytest.MonkeyPatch):
    """Test Generator.generate_job returns correct paths and calls correct functions and writes to params file."""
    monkeypatch.setattr(
        Generator, "_generate_run_path", lambda self, job, job_index: "/tmp_run"
    )
    log_path = generator_instance.root / "tmp_log"
    os.mkdir(log_path)
    monkeypatch.setattr(
        Generator, "_generate_log_path", lambda self, job, job_index: log_path
    )
    monkeypatch.setattr(Generator, "_output_files", lambda self, x, y: ("out", "err"))
    monkeypatch.setattr(
        Generator, "_build_operations", lambda self, job, job_path: None
    )
    job_path, out_file, err_file = generator_instance.generate_job(
        MockJob(), MockJob().index
    )
    assert job_path == "/tmp_run"
    assert out_file == "out"
    assert err_file == "err"
    with open(log_path / "smartsim_params.txt", "r") as file:
        content = file.read()
        assert "Generation start date and time:" in content


# INTEGRATED TESTS


def test_exp_private_generate_method(wlmutils, test_dir, generator_instance):
    """Test that Job directory was created from Experiment._generate."""
    # Create Experiment
    exp = Experiment(name="experiment_name", exp_path=test_dir)
    # Create Job
    app = Application("name", "python")
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job = Job(app, launch_settings)
    # Generate Job directory
    job_index = 1
    job_execution_path, _, _ = exp._generate(generator_instance, job, job_index)
    # Assert Job run directory exists
    assert osp.isdir(job_execution_path)
    # Assert Job log directory exists
    head, _ = os.path.split(job_execution_path)
    expected_log_path = pathlib.Path(head) / "log"
    assert osp.isdir(expected_log_path)


def test_exp_private_generate_method_ensemble(test_dir, wlmutils, generator_instance):
    """Test that Job directory was created from Experiment."""
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.as_jobs(launch_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    for i, job in enumerate(job_list):
        job_run_path, _, _ = exp._generate(generator_instance, job, i)
        head, _ = os.path.split(job_run_path)
        expected_log_path = pathlib.Path(head) / "log"
        assert osp.isdir(job_run_path)
        assert osp.isdir(pathlib.Path(expected_log_path))


def test_generate_ensemble_directory(wlmutils, generator_instance):
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.as_jobs(launch_settings)
    for i, job in enumerate(job_list):
        # Call Generator.generate_job
        path, _, _ = generator_instance.generate_job(job, i)
        # Assert run directory created
        assert osp.isdir(path)
        # Assert smartsim params file created
        head, _ = os.path.split(path)
        expected_log_path = pathlib.Path(head) / "log"
        assert osp.isdir(expected_log_path)
        assert osp.isfile(osp.join(expected_log_path, "smartsim_params.txt"))
        # Assert smartsim params correctly written to
        with open(expected_log_path / "smartsim_params.txt", "r") as file:
            content = file.read()
            assert "Generation start date and time:" in content


def test_generate_ensemble_directory_start(test_dir, wlmutils, monkeypatch):
    monkeypatch.setattr(
        "smartsim._core.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env, out, err: random_id(),
    )
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.as_jobs(launch_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    exp.start(*job_list)
    run_dir = listdir(test_dir)
    jobs_dir = os.path.join(test_dir, run_dir[0], "jobs")
    job_dir = listdir(jobs_dir)
    for ensemble_dir in job_dir:
        run_path = os.path.join(jobs_dir, ensemble_dir, "run")
        log_path = os.path.join(jobs_dir, ensemble_dir, "log")
        assert osp.isdir(run_path)
        assert osp.isdir(log_path)
    ids.clear()


def test_generate_ensemble_copy(test_dir, wlmutils, monkeypatch, get_gen_copy_dir):
    monkeypatch.setattr(
        "smartsim._core.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env, out, err: random_id(),
    )
    ensemble = Ensemble(
        "ensemble-name", "echo", replicas=2, files=EntityFiles(copy=get_gen_copy_dir)
    )
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.as_jobs(launch_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    exp.start(*job_list)
    run_dir = listdir(test_dir)
    jobs_dir = os.path.join(test_dir, run_dir[0], "jobs")
    job_dir = listdir(jobs_dir)
    for ensemble_dir in job_dir:
        copy_folder_path = os.path.join(jobs_dir, ensemble_dir, "run", "to_copy_dir")
        assert osp.isdir(copy_folder_path)
    ids.clear()


def test_generate_ensemble_symlink(
    test_dir, wlmutils, monkeypatch, get_gen_symlink_dir
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
    job_list = ensemble.as_jobs(launch_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    exp.start(*job_list)
    run_dir = listdir(test_dir)
    jobs_dir = os.path.join(test_dir, run_dir[0], "jobs")
    job_dir = listdir(jobs_dir)
    for ensemble_dir in job_dir:
        sym_file_path = pathlib.Path(jobs_dir) / ensemble_dir / "run" / "to_symlink_dir"
        assert osp.isdir(sym_file_path)
        assert sym_file_path.is_symlink()
        assert os.fspath(sym_file_path.resolve()) == osp.realpath(get_gen_symlink_dir)
    ids.clear()


def test_generate_ensemble_configure(
    test_dir, wlmutils, monkeypatch, get_gen_configure_dir
):
    monkeypatch.setattr(
        "smartsim._core.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env, out, err: random_id(),
    )
    params = {"PARAM0": [0, 1], "PARAM1": [2, 3]}
    # Retrieve a list of files for configuration
    tagged_files = sorted(glob(get_gen_configure_dir + "/*"))
    ensemble = Ensemble(
        "ensemble-name",
        "echo",
        replicas=1,
        files=EntityFiles(tagged=tagged_files),
        file_parameters=params,
    )
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.as_jobs(launch_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    id = exp.start(*job_list)
    run_dir = listdir(test_dir)
    jobs_dir = os.path.join(test_dir, run_dir[0], "jobs")

    def _check_generated(param_0, param_1, dir):
        assert osp.isdir(dir)
        assert osp.isfile(osp.join(dir, "tagged_0.sh"))
        assert osp.isfile(osp.join(dir, "tagged_1.sh"))

        with open(osp.join(dir, "tagged_0.sh")) as f:
            line = f.readline()
            assert line.strip() == f'echo "Hello with parameter 0 = {param_0}"'

        with open(osp.join(dir, "tagged_1.sh")) as f:
            line = f.readline()
            assert line.strip() == f'echo "Hello with parameter 1 = {param_1}"'

    _check_generated(0, 3, os.path.join(jobs_dir, "ensemble-name-1-1", "run"))
    _check_generated(1, 2, os.path.join(jobs_dir, "ensemble-name-2-2", "run"))
    _check_generated(1, 3, os.path.join(jobs_dir, "ensemble-name-3-3", "run"))
    _check_generated(0, 2, os.path.join(jobs_dir, "ensemble-name-0-0", "run"))
    ids.clear()
