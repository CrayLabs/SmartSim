from os import path as osp

from smartsim import Experiment
from smartsim._core.control.manifest import Manifest
from smartsim._core.generation.generator import Generator
from smartsim.entity._new_ensemble import Ensemble
from smartsim.entity.model import Application
from smartsim.launchable.job import Job
from smartsim.settings.launchSettings import LaunchSettings


def test_private_experiment_generation(test_dir, wlmutils):
    exp = Experiment(
        "test_exp", exp_path=test_dir, launcher=wlmutils.get_test_launcher()
    )
    manifest = Manifest()
    exp._generate(manifest)
    log_file = osp.join(test_dir, "smartsim_params.txt")
    assert osp.isfile(log_file)


def test_generate_experiment_directory(test_dir):
    manifest = Manifest()
    generate = Generator(test_dir, manifest)
    generate.generate_experiment()
    log_file = osp.join(test_dir, "smartsim_params.txt")
    assert osp.isfile(log_file)


def test_generate_application_directory(test_dir, wlmutils):
    # TODO remove run_settings and exe requirements
    path_1 = osp.join(test_dir, "app_folder_1")
    path_2 = osp.join(test_dir, "app_folder_2")
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    application_1 = Application(
        "app", exe="python", run_settings="RunSettings", path=path_1
    )
    application_2 = Application(
        "app", exe="python", run_settings="RunSettings", path=path_2
    )
    app_job_1 = Job(application_1, launch_settings)
    app_job_2 = Job(application_2, launch_settings)
    manifest = Manifest(app_job_1, app_job_2)
    generate = Generator(test_dir, manifest)
    generate.generate_experiment()
    assert osp.isdir(path_1)
    assert osp.isdir(path_2)


def test_generate_ensemble_directory(test_dir, wlmutils):
    path = osp.join(test_dir, "test")
    launch_settings = LaunchSettings(wlmutils.get_test_launcher())
    jobs = Ensemble("ensemble", "python", replicas=9, path=path).as_jobs(
        launch_settings
    )
    manifest = Manifest(jobs)
    generate = Generator(test_dir, manifest)
    generate.generate_experiment()
    assert osp.isdir(path)
    for i in range(9):
        assert osp.isdir(osp.join(path, "ensemble-" + str(i)))


def test_to_copy_operation(fileutils, wlmutils, test_dir):
    path_1 = osp.join(test_dir, "app_folder_3")
    application_1 = Application(
        "app", exe="python", run_settings="RunSettings", path=path_1
    )
    script = fileutils.get_test_conf_path("sleep.py")
    application_1.attach_generator_files(to_copy=script)
    job = Job(application_1, LaunchSettings(wlmutils.get_test_launcher()))
    manifest = Manifest(job)
    generate = Generator(test_dir, manifest)
    generate.generate_experiment()
    output = generate.build_operations(application_1)
    assert output == [["temporary", "copy"]]


def get_gen_file(fileutils, filename):
    return fileutils.get_test_conf_path(osp.join("generator_files", filename))


def test_to_link_operation(fileutils, wlmutils, test_dir):
    path_1 = osp.join(test_dir, "app_folder_3")
    application_1 = Application(
        "app", exe="python", run_settings="RunSettings", path=path_1
    )
    symlink_dir = get_gen_file(fileutils, "to_symlink_dir")
    application_1.attach_generator_files(to_symlink=symlink_dir)
    job = Job(application_1, LaunchSettings(wlmutils.get_test_launcher()))
    manifest = Manifest(job)
    generate = Generator(test_dir, manifest)
    generate.generate_experiment()
    output = generate.build_operations(application_1)
    assert output == [["temporary", "link"]]


def test_to_configure_operation(fileutils, test_dir, wlmutils):
    path_1 = osp.join(test_dir, "app_folder_3")
    application_1 = Application(
        "app", exe="python", run_settings="RunSettings", path=path_1
    )
    config = get_gen_file(fileutils, "in.atm")
    application_1.attach_generator_files(to_configure=config)
    job = Job(application_1, LaunchSettings(wlmutils.get_test_launcher()))
    manifest = Manifest(job)
    generate = Generator(test_dir, manifest)
    generate.generate_experiment()
    output = generate.build_operations(application_1)
    assert output == [["temporary", "configure"]]
