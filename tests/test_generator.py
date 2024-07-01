from smartsim.settings.launchSettings import LaunchSettings
from smartsim.entity._new_ensemble import Ensemble
from smartsim.entity.model import Application
from smartsim._core.control.manifest import Manifest
from smartsim._core.generation.generator import Generator
from smartsim.launchable.job import Job
from os import path as osp
import os

launch_settings = LaunchSettings("slurm")

# TODO remove run_settings and exe requirements
application = Application("app", exe="python",run_settings="RunSettings")

def test_generate_experiment_directory(test_dir):
    manifest = Manifest()
    generate = Generator(test_dir, manifest)
    generate.generate_experiment()
    log_file = osp.join(test_dir, "smartsim_params.txt")
    assert osp.isfile(log_file)

def test_generate_ensemble_directory(test_dir):
    manifest = Manifest()
    generate = Generator(test_dir, manifest)
    generate.generate_experiment()
    log_file = osp.join(test_dir, "smartsim_params.txt")
    assert osp.isfile(log_file)