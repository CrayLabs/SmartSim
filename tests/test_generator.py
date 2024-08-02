import filecmp
import os
import pathlib
from glob import glob
from logging import DEBUG, INFO
from os import environ, listdir
from os import path as osp

import pytest

from smartsim import Experiment
from smartsim._core.generation.generator import Generator
from smartsim.entity import Application, Ensemble, SmartSimEntity, _mock
from smartsim.launchable import Job
from smartsim.settings.launchSettings import LaunchSettings

# TODO Test ensemble copy, config, symlink when Ensemble.attach_generator_files added
# TODO Add JobGroup tests when JobGroup becomes a Launchable

pytestmark = pytest.mark.group_a

@pytest.fixture
def get_gen_copy_file(fileutils):
    return fileutils.get_test_conf_path(osp.join("generator_files", "to_copy_dir"))

@pytest.fixture
def get_gen_symlink_file(fileutils):
    return fileutils.get_test_conf_path(osp.join("generator_files", "to_symlink_dir"))

# Mock Launcher
class NoOpLauncher:
    @classmethod
    def create(cls, _):
        return cls()

    def start(self, _):
        return "anything"


@pytest.fixture
def echo_app():
    yield SmartSimEntity("echo_app", run_settings=_mock.Mock())


@pytest.fixture
def generator_instance(test_dir) -> Generator:
    """Fixture to create an instance of Generator."""
    experiment_path = osp.join(test_dir, "experiment_name")
    return Generator(exp_path=experiment_path, run_id="mock_run")


@pytest.fixture
def job_instance(wlmutils, echo_app) -> Job:
    """Fixture to create an instance of Job."""
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job = Job(echo_app, launch_settings)
    return job

def test_log_file_path(generator_instance):
    """Test if the log_file property returns the correct path."""
    path = "/tmp"
    expected_path = osp.join(path, "smartsim_params.txt")
    assert generator_instance.log_file(path) == expected_path


def test_generate_job_directory(test_dir, wlmutils):
    """Test Generator.generate_job"""
    # Experiment path
    experiment_path = osp.join(test_dir, "experiment_name")
    # Create Job
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("app_name", exe="python", run_settings="RunSettings")
    job = Job(app, launch_settings)
    # Mock start id
    run_id = "mock_run"
    # Generator instance
    gen = Generator(exp_path=experiment_path, run_id=run_id)
    # Call Generator.generate_job
    job_path = gen.generate_job(job)
    assert isinstance(job_path, pathlib.Path)
    expected_run_path = (
        pathlib.Path(experiment_path)
        / run_id
        / f"{job.__class__.__name__.lower()}s"
        / app.name
        / "run"
    )
    assert job_path == expected_run_path
    expected_log_path = (
        pathlib.Path(experiment_path)
        / run_id
        / f"{job.__class__.__name__.lower()}s"
        / app.name
        / "log"
    )
    assert osp.isdir(expected_run_path)
    assert osp.isdir(expected_log_path)
    assert osp.isfile(osp.join(expected_log_path, "smartsim_params.txt"))


# def test_exp_private_generate_method_app(test_dir, job_instance):
#     """Test that Job directory was created from Experiment."""
#     no_op_exp = Experiment(name="No-Op-Exp", exp_path=test_dir)
#     job_execution_path = no_op_exp._generate(job_instance)
#     assert osp.isdir(job_execution_path)
#     head, _ = os.path.split(job_execution_path)
#     expected_log_path = pathlib.Path(head) / "log"
#     assert osp.isdir(expected_log_path)
#     assert osp.isfile(osp.join(job_execution_path, "smartsim_params.txt"))


def test_generate_copy_file(fileutils, wlmutils, test_dir):
    # Create the Job and attach generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", "RunSettings")
    script = fileutils.get_test_conf_path("sleep.py")
    app.attach_generator_files(to_copy=script)
    job = Job(app, launch_settings)

    # Create the experiment
    experiment_path = osp.join(test_dir, "experiment_name")
    gen = Generator(exp_path=experiment_path, run_id="temp_run")
    path = gen.generate_job(job)
    expected_file = pathlib.Path(path) / "sleep.py"
    assert osp.isfile(expected_file)


def test_generate_copy_directory(wlmutils, test_dir, get_gen_copy_file):
    # Create the Job and attach generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", "RunSettings")
    app.attach_generator_files(to_copy=get_gen_copy_file)
    job = Job(app, launch_settings)

    # Create the experiment
    experiment_path = osp.join(test_dir, "experiment_name")
    gen = Generator(exp_path=experiment_path, run_id="temp_run")
    path = gen.generate_job(job)
    expected_file = pathlib.Path(path) / "mock.txt"
    assert osp.isfile(expected_file)


def test_generate_symlink_directory(wlmutils, test_dir, get_gen_symlink_file):
    # Create the Job and attach generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", "RunSettings")
    # Path of directory to symlink
    symlink_dir = get_gen_symlink_file
    # Attach directory to Application
    app.attach_generator_files(to_symlink=symlink_dir)
    # Create Job
    job = Job(app, launch_settings)

    # Create the experiment
    experiment_path = osp.join(test_dir, "experiment_name")
    gen = Generator(exp_path=experiment_path, run_id="temp_run")
    # Generate Experiment file structure
    job_path = gen.generate_job(job)
    expected_folder = pathlib.Path(job_path) / "to_symlink_dir"
    assert osp.isdir(expected_folder)
    # Combine symlinked file list and original file list for comparison
    for written, correct in zip(listdir(symlink_dir), listdir(expected_folder)):
        # For each pair, check if the filenames are equal
        assert written == correct


def test_generate_symlink_file(get_gen_symlink_file, wlmutils, test_dir):
    # Create the Job and attach generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", "RunSettings")
    # Path of directory to symlink
    symlink_dir = get_gen_symlink_file
    # Get a list of all files in the directory
    symlink_files = sorted(glob(symlink_dir + "/*"))
    # Attach directory to Application
    app.attach_generator_files(to_symlink=symlink_files)
    # Create Job
    job = Job(app, launch_settings)
    # Create the experiment
    experiment_path = osp.join(test_dir, "experiment_name")
    gen = Generator(exp_path=experiment_path, run_id="mock_run")
    # Generate Experiment file structure
    job_path = gen.generate_job(job)
    expected_file = pathlib.Path(job_path) / "mock2.txt"
    assert osp.isfile(expected_file)


def test_generate_configure(fileutils, wlmutils, test_dir):
    # Directory of files to configure
    conf_path = fileutils.get_test_conf_path(
        osp.join("generator_files", "easy", "marked/")
    )
    # Retrieve a list of files for configuration
    tagged_files = sorted(glob(conf_path + "/*"))
    # Retrieve directory of files to compare after Experiment.generate_experiment completion
    correct_path = fileutils.get_test_conf_path(
        osp.join("generator_files", "easy", "correct/")
    )
    # Retrieve list of files in correctly tagged directory for comparison
    correct_files = sorted(glob(correct_path + "/*"))
    # Initialize a Job
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    param_dict = {
        "5": 10,
        "FIRST": "SECOND",
        "17": 20,
        "65": "70",
        "placeholder": "group leftupper region",
        "1200": "120",
        "VALID": "valid",
    }
    app = Application("name_1", "python", "RunSettings", params=param_dict)
    app.attach_generator_files(to_configure=tagged_files)
    job = Job(app, launch_settings)

    # Spin up Experiment
    experiment_path = osp.join(test_dir, "experiment_name")
    # Spin up Generator
    gen = Generator(exp_path=experiment_path, run_id="temp_run")
    # Execute file generation
    job_path = gen.generate_job(job)
    # Retrieve the list of configured files in the test directory
    configured_files = sorted(glob(str(job_path) + "/*"))
    # Use filecmp.cmp to check that the corresponding files are equal
    for written, correct in zip(configured_files, correct_files):
        assert filecmp.cmp(written, correct)
    # Validate that log file exists
   #  assert osp.isdir()
    # Validate that smartsim params files exists
    # smartsim_params_path = osp.join(job_path, "smartsim_params.txt")
    # assert osp.isfile(smartsim_params_path)


def test_exp_private_generate_method_ensemble(test_dir, wlmutils, generator_instance):
    """Test that Job directory was created from Experiment."""
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.as_jobs(launch_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    for job in job_list:
        job_execution_path = exp._generate(generator_instance, job)
        head, _ = os.path.split(job_execution_path)
        expected_log_path = pathlib.Path(head) / "log"
        assert osp.isdir(job_execution_path)
        assert osp.isdir(pathlib.Path(expected_log_path))


def test_generate_ensemble_directory(test_dir, wlmutils, generator_instance):
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.as_jobs(launch_settings)
    for job in job_list:
        job_path = generator_instance.generate_job(job)
        assert osp.isdir(job_path)