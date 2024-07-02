from smartsim.settings.launchSettings import LaunchSettings
from smartsim.entity._new_ensemble import Ensemble
from smartsim.entity.model import Application
from smartsim._core.control.manifest import Manifest
from smartsim._core.generation.generator import Generator
from smartsim.launchable.job import Job
from os import path as osp
import os

def test_generate_experiment_directory(test_dir):
    manifest = Manifest()
    generate = Generator(test_dir, manifest)
    generate.generate_experiment()
    log_file = osp.join(test_dir, "smartsim_params.txt")
    assert osp.isfile(log_file)

def test_generate_application_directory(test_dir):
    # TODO remove run_settings and exe requirements
    path_1 = osp.join(test_dir, "app_folder_1")
    path_2 = osp.join(test_dir, "app_folder_2")
    launch_settings = LaunchSettings("slurm")
    application_1 = Application("app", exe="python",run_settings="RunSettings", path=path_1)
    application_2 = Application("app", exe="python",run_settings="RunSettings", path=path_2)
    app_job_1 = Job(application_1, launch_settings)
    app_job_2 = Job(application_2, launch_settings)
    manifest = Manifest(app_job_1, app_job_2)
    generate = Generator(test_dir, manifest)
    generate.generate_experiment()
    assert osp.isdir(path_1)
    assert osp.isdir(path_2)

def test_generate_ensemble_directory(test_dir):
    path_1 = osp.join(test_dir, "ensemble")
    launch_settings = LaunchSettings("slurm")
    jobs = Ensemble("ensemble", "python", replicas=2, path=path_1).as_jobs(launch_settings)
    manifest = Manifest(jobs)
    generate = Generator(test_dir, manifest)
    generate.generate_experiment()
    
    