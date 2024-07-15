from os import path as osp

from smartsim import Experiment
from smartsim._core.control.manifest import Manifest
from smartsim._core.generation.generator import Generator
from smartsim.entity.model import Application
from smartsim.launchable.job import Job
from smartsim.settings.launchSettings import LaunchSettings


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
