import datetime
from logging import DEBUG, INFO
from os import environ
from os import path as osp

import pytest

from smartsim import Experiment
from smartsim._core.generation.generator import Generator
from smartsim.entity import Application, Ensemble
from smartsim.launchable import Job, JobGroup
from smartsim.settings.builders.launch import SlurmArgBuilder
from smartsim.settings.dispatch import Dispatcher
from smartsim.settings.launchSettings import LaunchSettings


class NoOpLauncher:
    @classmethod
    def create(cls, _):
        return cls()

    def start(self, _):
        return "anything"

class EchoApp:
    def as_program_arguments(self):
        return ["echo", "Hello", "World!"]


@pytest.fixture
def gen_instance_for_job(test_dir, wlmutils) -> Generator:
    """Fixture to create an instance of Generator."""
    experiment_path = osp.join(test_dir, "experiment_name")
    run_ID = (
        "run-"
        + datetime.datetime.now().strftime("%H:%M:%S")
        + "-"
        + datetime.datetime.now().strftime("%Y-%m-%d")
    )
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    application_1 = Application("app_name", exe="python", run_settings="RunSettings")
    job = Job(application_1, launch_settings)
    return Generator(gen_path=experiment_path, run_ID=run_ID, job=job)


@pytest.fixture
def job_group_instance(test_dir, wlmutils) -> Generator:
    """Fixture to create an instance of Generator."""
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    application_1 = Application("app_name_1", exe="python", run_settings="RunSettings")
    application_2 = Application("app_name_2", exe="python", run_settings="RunSettings")
    job_group = JobGroup(application_1, application_2, launch_settings)
    return job_group


@pytest.fixture
def job_instance(test_dir, wlmutils) -> Generator:
    """Fixture to create an instance of Generator."""
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job = Job(EchoApp(), launch_settings)
    return job


def test_default_log_level(gen_instance_for_job):
    """Test if the default log level is INFO."""
    assert gen_instance_for_job.log_level == INFO


def test_debug_log_level(gen_instance_for_job):
    """Test if the log level is DEBUG when environment variable is set to "debug"."""
    environ["SMARTSIM_LOG_LEVEL"] = "debug"
    assert gen_instance_for_job.log_level == DEBUG
    # Clean up: unset the environment variable
    # TODO might need to set it to info here?
    environ.pop("SMARTSIM_LOG_LEVEL", None)


def test_log_file_path(gen_instance_for_job):
    """Test if the log_file property returns the correct path."""
    expected_path = osp.join(gen_instance_for_job.path, "smartsim_params.txt")
    assert gen_instance_for_job.log_file == expected_path


def test_generate_job_directory(gen_instance_for_job):
    gen_instance_for_job.generate_experiment()
    assert osp.isdir(gen_instance_for_job.path)
    assert osp.isdir(gen_instance_for_job.log_path)


def test_full_exp_generate_job_directory(test_dir, job_instance):
    no_op_dispatch = Dispatcher()
    no_op_dispatch.dispatch(SlurmArgBuilder, to_launcher=NoOpLauncher)
    no_op_exp = Experiment(
        name="No-Op-Exp", exp_path=test_dir, settings_dispatcher=no_op_dispatch
    )
    job_execution_path = no_op_exp._generate(job_instance)
    assert osp.isdir(job_execution_path)

def test_generate_ensemble_directory(test_dir, wlmutils):
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    job_list = ensemble.as_jobs(launch_settings)
    for job in job_list:
        run_ID = "temp_run"
        gen = Generator(gen_path=test_dir, run_ID=run_ID, job=job)
        gen.generate_experiment()
        assert osp.isdir(gen.path)