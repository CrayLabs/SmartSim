from os import path as osp
from os import environ
import pytest

from smartsim import Experiment
from smartsim._core.control.manifest import Manifest
from smartsim._core.generation.generator import Generator
from smartsim.entity.model import Application
from smartsim.launchable.job import Job
from smartsim.settings.launchSettings import LaunchSettings
from logging import DEBUG, INFO

@pytest.fixture
def generator_instance(test_dir, wlmutils) -> Generator:
    """Fixture to create an instance of Generator."""
    experiment_path = osp.join(test_dir, "experiment_name")
    app_path = osp.join(experiment_path, "app_name")
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    application_1 = Application(
        "app_name", exe="python", run_settings="RunSettings", path=app_path
    )
    job = Job(application_1, launch_settings)
    return Generator(gen_path=experiment_path, job=job)

def test_default_log_level(generator_instance):
    """Test if the default log level is INFO."""
    assert generator_instance.log_level == INFO

def test_debug_log_level(generator_instance):
    """Test if the log level is DEBUG when environment variable is set to "debug"."""
    environ["SMARTSIM_LOG_LEVEL"] = "debug"
    assert generator_instance.log_level == DEBUG
    # Clean up: unset the environment variable
    environ.pop("SMARTSIM_LOG_LEVEL", None)

def test_generate_custom_id(generator_instance):
    """Test if the generated custom ID has the correct length."""
    custom_id = generator_instance._generate_custom_id()
    assert len(custom_id) == 9
    assert custom_id.startswith("-")
    # assert custom_id[1:].isalnum()

def test_experiment_directory(test_dir, wlmutils):
    # TODO remove run_settings and exe requirements
    experiment_path = osp.join(test_dir, "experiment_name")
    app_path = osp.join(experiment_path, "app_name")
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    application_1 = Application(
        "app_name", exe="python", run_settings="RunSettings", path=app_path
    )
    job = Job(application_1, launch_settings)
    generator = Generator(gen_path=experiment_path, job=job)
    print("here")
    print(generator.run_path)
    # generator.generate_experiment()
    # assert osp.isdir(experiment_path)
    # assert osp.isdir(app_path)
