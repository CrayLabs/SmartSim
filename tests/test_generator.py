import filecmp
import itertools
import os
import pathlib
from glob import glob
from os import listdir
from os import path as osp

import pytest

from smartsim import Experiment
from smartsim._core.generation.generator import Generator
from smartsim.entity import Application, Ensemble, SmartSimEntity, _mock
from smartsim.entity.files import EntityFiles
from smartsim.launchable import Job
from smartsim.settings import LaunchSettings, dispatch

# TODO Add JobGroup tests when JobGroup becomes a Launchable

pytestmark = pytest.mark.group_a


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
    yield Generator(root=root)


def test_log_file_path(generator_instance):
    """Test if the log_file function returns the correct log path."""
    base_path = "/tmp"
    expected_path = osp.join(base_path, "smartsim_params.txt")
    assert generator_instance.log_file(base_path) == pathlib.Path(expected_path)


def test_generate_job_directory(wlmutils, generator_instance):
    """Test Generator.generate_job"""
    # Create Job
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application(
        "app_name", exe="python", run_settings="RunSettings"
    )  # Mock RunSettings
    job = Job(app, launch_settings)
    # Mock id
    run_id = "mock_run"
    # Create run directory
    run_path = generator_instance.root / "run"
    run_path.mkdir(parents=True)
    assert osp.isdir(run_path)
    # Create log directory
    log_path = generator_instance.root / "log"
    log_path.mkdir(parents=True)
    assert osp.isdir(log_path)
    # Call Generator.generate_job
    generator_instance.generate_job(job, run_path, log_path)
    # Assert smartsim params file created
    assert osp.isfile(osp.join(log_path, "smartsim_params.txt"))
    # Assert smartsim params correctly written to
    with open(log_path / "smartsim_params.txt", "r") as file:
        content = file.read()
        assert "Generation start date and time:" in content


def test_exp_private_generate_method(wlmutils, test_dir, generator_instance):
    """Test that Job directory was created from Experiment._generate."""
    # Create Experiment
    exp = Experiment(name="experiment_name", exp_path=test_dir)
    # Create Job
    app = Application("name", "python", run_settings="RunSettings")  # Mock RunSettings
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job = Job(app, launch_settings)
    # Generate Job directory
    job_index = 1
    job_execution_path = exp._generate(generator_instance, job, job_index)
    # Assert Job run directory exists
    assert osp.isdir(job_execution_path)
    # Assert Job log directory exists
    head, _ = os.path.split(job_execution_path)
    expected_log_path = pathlib.Path(head) / "log"
    assert osp.isdir(expected_log_path)


def test_generate_copy_file(fileutils, wlmutils, generator_instance):
    """Test that attached copy files are copied into Job directory"""
    # Create the Job and attach copy generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", run_settings="RunSettings")  # Mock RunSettings
    script = fileutils.get_test_conf_path("sleep.py")
    app.attach_generator_files(to_copy=script)
    job = Job(app, launch_settings)

    # Call Generator.generate_job
    run_path = generator_instance.root / "run"
    run_path.mkdir(parents=True)
    log_path = generator_instance.root / "log"
    log_path.mkdir(parents=True)
    generator_instance.generate_job(job, run_path, log_path)
    expected_file = run_path / "sleep.py"
    assert osp.isfile(expected_file)


def test_generate_copy_directory(wlmutils, get_gen_copy_dir, generator_instance):
    # Create the Job and attach generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", run_settings="RunSettings")  # Mock RunSettings
    app.attach_generator_files(to_copy=get_gen_copy_dir)
    job = Job(app, launch_settings)

    # Call Generator.generate_job
    run_path = generator_instance.root / "run"
    run_path.mkdir(parents=True)
    log_path = generator_instance.root / "log"
    log_path.mkdir(parents=True)
    generator_instance.generate_job(job, run_path, log_path)
    expected_folder = run_path / "to_copy_dir"
    assert osp.isdir(expected_folder)


def test_generate_symlink_directory(wlmutils, generator_instance, get_gen_symlink_dir):
    # Create the Job and attach generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", run_settings="RunSettings")  # Mock RunSettings
    # Attach directory to Application
    app.attach_generator_files(to_symlink=get_gen_symlink_dir)
    # Create Job
    job = Job(app, launch_settings)

    # Call Generator.generate_job
    run_path = generator_instance.root / "run"
    run_path.mkdir(parents=True)
    log_path = generator_instance.root / "log"
    log_path.mkdir(parents=True)
    generator_instance.generate_job(job, run_path, log_path)
    expected_folder = run_path / "to_symlink_dir"
    assert osp.isdir(expected_folder)
    assert expected_folder.is_symlink()
    assert os.fspath(expected_folder.resolve()) == osp.realpath(get_gen_symlink_dir)
    # Combine symlinked file list and original file list for comparison
    for written, correct in itertools.zip_longest(
        listdir(get_gen_symlink_dir), listdir(expected_folder)
    ):
        # For each pair, check if the filenames are equal
        assert written == correct


def test_generate_symlink_file(get_gen_symlink_dir, wlmutils, generator_instance):
    # Create the Job and attach generator file
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    app = Application("name", "python", "RunSettings")
    # Path of directory to symlink
    symlink_dir = get_gen_symlink_dir
    # Get a list of all files in the directory
    symlink_files = sorted(glob(symlink_dir + "/*"))
    # Attach directory to Application
    app.attach_generator_files(to_symlink=symlink_files)
    # Create Job
    job = Job(app, launch_settings)

    # Call Generator.generate_job
    run_path = generator_instance.root / "run"
    run_path.mkdir(parents=True)
    log_path = generator_instance.root / "log"
    log_path.mkdir(parents=True)
    generator_instance.generate_job(job, run_path, log_path)
    expected_file = pathlib.Path(run_path) / "mock2.txt"
    assert osp.isfile(expected_file)
    assert expected_file.is_symlink()
    assert os.fspath(expected_file.resolve()) == osp.join(
        osp.realpath(get_gen_symlink_dir), "mock2.txt"
    )


def test_generate_configure(fileutils, wlmutils, generator_instance):
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

    # Call Generator.generate_job
    run_path = generator_instance.root / "run"
    run_path.mkdir(parents=True)
    log_path = generator_instance.root / "log"
    log_path.mkdir(parents=True)
    generator_instance.generate_job(job, run_path, log_path)
    # Retrieve the list of configured files in the test directory
    configured_files = sorted(glob(str(run_path) + "/*"))
    # Use filecmp.cmp to check that the corresponding files are equal
    for written, correct in itertools.zip_longest(configured_files, correct_files):
        assert filecmp.cmp(written, correct)


def test_exp_private_generate_method_ensemble(test_dir, wlmutils, generator_instance):
    """Test that Job directory was created from Experiment."""
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.as_jobs(launch_settings)
    exp = Experiment(name="exp_name", exp_path=test_dir)
    for i, job in enumerate(job_list):
        job_run_path = exp._generate(generator_instance, job, i)
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
        run_path = generator_instance.root / f"run-{i}"
        run_path.mkdir(parents=True)
        log_path = generator_instance.root / f"log-{i}"
        log_path.mkdir(parents=True)
        generator_instance.generate_job(job, run_path, log_path)
        # Assert smartsim params file created
        assert osp.isfile(osp.join(log_path, "smartsim_params.txt"))
        # Assert smartsim params correctly written to
        with open(log_path / "smartsim_params.txt", "r") as file:
            content = file.read()
            assert "Generation start date and time:" in content


def test_generate_ensemble_directory_start(test_dir, wlmutils, monkeypatch):
    monkeypatch.setattr(
        "smartsim.settings.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env: "exit",
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


def test_generate_ensemble_copy(test_dir, wlmutils, monkeypatch, get_gen_copy_dir):
    monkeypatch.setattr(
        "smartsim.settings.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env: "exit",
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


def test_generate_ensemble_symlink(
    test_dir, wlmutils, monkeypatch, get_gen_symlink_dir
):
    monkeypatch.setattr(
        "smartsim.settings.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env: "exit",
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


def test_generate_ensemble_configure(
    test_dir, wlmutils, monkeypatch, get_gen_configure_dir
):
    monkeypatch.setattr(
        "smartsim.settings.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env: "exit",
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
    exp.start(*job_list)
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
