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
from smartsim.entity import Application, Ensemble
from smartsim.launchable import Job
from smartsim.settings.launchSettings import LaunchSettings

# TODO test ensemble copy, config, symlink when ensemble.attach_generator_files added
# TODO remove ensemble tests and replace with JobGroup when start jobgroup is supported

pytestmark = pytest.mark.group_a


def get_gen_file(fileutils, filename):
    return fileutils.get_test_conf_path(osp.join("generator_files", filename))


# Mock Launcher
class NoOpLauncher:
    @classmethod
    def create(cls, _):
        return cls()

    def start(self, _):
        return "anything"


# Mock Application
class EchoApp:
    name = "echo_app"

    def as_program_arguments(self):
        return ["echo", "Hello", "World!"]


@pytest.fixture
def generator_instance(test_dir, wlmutils) -> Generator:
    """Fixture to create an instance of Generator."""
    experiment_path = osp.join(test_dir, "experiment_name")
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("app_name", exe="python", run_settings="RunSettings")
    job = Job(app, launch_settings)
    return Generator(gen_path=experiment_path, run_ID="mock_run", job=job)


@pytest.fixture
def job_instance(wlmutils) -> Job:
    """Fixture to create an instance of Job."""
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job = Job(EchoApp(), launch_settings)
    return job


def test_default_log_level(generator_instance):
    """Test if the default log level is INFO."""
    assert generator_instance.log_level == INFO


def test_debug_log_level(generator_instance):
    """Test if the log level is DEBUG when environment variable is set to "debug"."""
    environ["SMARTSIM_LOG_LEVEL"] = "debug"
    assert generator_instance.log_level == DEBUG
    # Clean up: unset the environment variable
    environ.pop("SMARTSIM_LOG_LEVEL", None)


def test_log_file_path(generator_instance):
    """Test if the log_file property returns the correct path."""
    expected_path = osp.join(generator_instance.path, "smartsim_params.txt")
    assert generator_instance.log_file == expected_path


def test_generate_job_directory(test_dir, wlmutils):
    """Test that Job directory was created."""
    experiment_path = osp.join(test_dir, "experiment_name")
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("app_name", exe="python", run_settings="RunSettings")
    job = Job(app, launch_settings)
    run_ID = "mock_run"
    gen = Generator(gen_path=experiment_path, run_ID=run_ID, job=job)
    gen.generate_experiment()
    expected_run_path = (
        pathlib.Path(experiment_path)
        / run_ID
        / f"{job.__class__.__name__.lower()}s"
        / app.name
        / "run"
    )
    expected_log_path = (
        pathlib.Path(experiment_path)
        / run_ID
        / f"{job.__class__.__name__.lower()}s"
        / app.name
        / "log"
    )
    assert gen.path == str(expected_run_path)
    assert gen.log_path == str(expected_log_path)
    assert osp.isdir(expected_run_path)
    assert osp.isdir(expected_log_path)
    assert osp.isfile(osp.join(expected_run_path, "smartsim_params.txt"))


def test_exp_private_generate_method_app(test_dir, job_instance):
    """Test that Job directory was created from Experiment."""
    no_op_exp = Experiment(name="No-Op-Exp", exp_path=test_dir)
    job_execution_path = no_op_exp._generate(job_instance)
    assert osp.isdir(job_execution_path)
    head, _ = os.path.split(job_execution_path)
    expected_log_path = pathlib.Path(head) / "log"
    assert osp.isdir(expected_log_path)
    assert osp.isfile(osp.join(job_execution_path, "smartsim_params.txt"))


def test_generate_copy_file(fileutils, wlmutils, test_dir):
    # Create the Job and attach generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", "RunSettings")
    script = fileutils.get_test_conf_path("sleep.py")
    app.attach_generator_files(to_copy=script)
    job = Job(app, launch_settings)

    # Create the experiment
    experiment_path = osp.join(test_dir, "experiment_name")
    gen = Generator(gen_path=experiment_path, run_ID="temp_run", job=job)
    path = gen.generate_experiment()
    expected_file = pathlib.Path(path) / "sleep.py"
    assert osp.isfile(expected_file)


def test_generate_copy_directory(fileutils, wlmutils, test_dir):
    # Create the Job and attach generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", "RunSettings")
    copy_dir = get_gen_file(fileutils, "to_copy_dir")
    app.attach_generator_files(to_copy=copy_dir)
    job = Job(app, launch_settings)

    # Create the experiment
    experiment_path = osp.join(test_dir, "experiment_name")
    gen = Generator(gen_path=experiment_path, run_ID="temp_run", job=job)
    gen.generate_experiment()
    expected_file = pathlib.Path(gen.path) / "mock.txt"
    assert osp.isfile(expected_file)


def test_generate_symlink_directory(fileutils, wlmutils, test_dir):
    # Create the Job and attach generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", "RunSettings")
    # Path of directory to symlink
    symlink_dir = get_gen_file(fileutils, "to_symlink_dir")
    # Attach directory to Application
    app.attach_generator_files(to_symlink=symlink_dir)
    # Create Job
    job = Job(app, launch_settings)

    # Create the experiment
    experiment_path = osp.join(test_dir, "experiment_name")
    gen = Generator(gen_path=experiment_path, run_ID="temp_run", job=job)
    # Generate Experiment file structure
    gen.generate_experiment()
    expected_folder = pathlib.Path(gen.path) / "to_symlink_dir"
    assert osp.isdir(expected_folder)
    # Combine symlinked file list and original file list for comparison
    for written, correct in zip(listdir(symlink_dir), listdir(expected_folder)):
        # For each pair, check if the filenames are equal
        assert written == correct


def test_generate_symlink_file(fileutils, wlmutils, test_dir):
    # Create the Job and attach generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", "RunSettings")
    # Path of directory to symlink
    symlink_dir = get_gen_file(fileutils, "to_symlink_dir")
    # Get a list of all files in the directory
    symlink_files = sorted(glob(symlink_dir + "/*"))
    # Attach directory to Application
    app.attach_generator_files(to_symlink=symlink_files)
    # Create Job
    job = Job(app, launch_settings)
    # Create the experiment
    experiment_path = osp.join(test_dir, "experiment_name")
    gen = Generator(gen_path=experiment_path, run_ID="test", job=job)
    # Generate Experiment file structure
    gen.generate_experiment()
    expected_file = pathlib.Path(gen.path) / "mock2.txt"
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
    gen = Generator(gen_path=experiment_path, run_ID="temp_run", job=job)
    # Execute file generation
    job_path = gen.generate_experiment()
    # Retrieve the list of configured files in the test directory
    configured_files = sorted(glob(job_path + "/*"))
    # Use filecmp.cmp to check that the corresponding files are equal
    for written, correct in zip(configured_files, correct_files):
        assert filecmp.cmp(written, correct)
    # Validate that log file exists
    assert osp.isdir(gen.log_path)
    # Validate that smartsim params files exists
    smartsim_params_path = osp.join(job_path, "smartsim_params.txt")
    assert osp.isfile(smartsim_params_path)


def test_exp_private_generate_method_ensemble(test_dir, wlmutils):
    """Test that Job directory was created from Experiment."""
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.as_jobs(launch_settings)
    no_op_exp = Experiment(name="No-Op-Exp", exp_path=test_dir)
    for job in job_list:
        job_execution_path = no_op_exp._generate(job)
        head, _ = os.path.split(job_execution_path)
        expected_log_path = pathlib.Path(head) / "log"
        assert osp.isdir(job_execution_path)
        assert osp.isdir(pathlib.Path(expected_log_path))


def test_generate_ensemble_directory(test_dir, wlmutils):
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.as_jobs(launch_settings)
    for job in job_list:
        run_ID = "temp_run"
        gen = Generator(gen_path=test_dir, run_ID=run_ID, job=job)
        gen.generate_experiment()
        assert osp.isdir(gen.path)
        assert osp.isdir(pathlib.Path(gen.log_path))
