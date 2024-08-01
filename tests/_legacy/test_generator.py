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
from os import path as osp

import pytest
from tabulate import tabulate

from smartsim import Experiment
from smartsim._core.generation import Generator
from smartsim.database import FeatureStore
from smartsim.settings import RunSettings

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


rs = RunSettings("python", exe_args="sleep.py")


"""
Test the generation of files and input data for an experiment

TODO
 - test lists of inputs for each file type
 - test empty directories
 - test re-generation

"""


def get_gen_file(fileutils, filename):
    return fileutils.get_test_conf_path(osp.join("generator_files", filename))


def test_ensemble(fileutils, test_dir):
    exp = Experiment("gen-test", launcher="local")

    gen = Generator(test_dir)
    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble("test", params=params, run_settings=rs)

    config = get_gen_file(fileutils, "in.atm")
    ensemble.attach_generator_files(to_configure=config)
    gen.generate_experiment(ensemble)

    assert len(ensemble) == 9
    assert osp.isdir(osp.join(test_dir, "test"))
    for i in range(9):
        assert osp.isdir(osp.join(test_dir, "test/test_" + str(i)))


def test_ensemble_overwrite(fileutils, test_dir):
    exp = Experiment("gen-test-overwrite", launcher="local")

    gen = Generator(test_dir, overwrite=True)

    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble("test", params=params, run_settings=rs)

    config = get_gen_file(fileutils, "in.atm")
    ensemble.attach_generator_files(to_configure=[config])
    gen.generate_experiment(ensemble)

    # re generate without overwrite
    config = get_gen_file(fileutils, "in.atm")
    ensemble.attach_generator_files(to_configure=[config])
    gen.generate_experiment(ensemble)

    assert len(ensemble) == 9
    assert osp.isdir(osp.join(test_dir, "test"))
    for i in range(9):
        assert osp.isdir(osp.join(test_dir, "test/test_" + str(i)))


def test_ensemble_overwrite_error(fileutils, test_dir):
    exp = Experiment("gen-test-overwrite-error", launcher="local")

    gen = Generator(test_dir)

    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble("test", params=params, run_settings=rs)

    config = get_gen_file(fileutils, "in.atm")
    ensemble.attach_generator_files(to_configure=[config])
    gen.generate_experiment(ensemble)

    # re generate without overwrite
    config = get_gen_file(fileutils, "in.atm")
    ensemble.attach_generator_files(to_configure=[config])
    with pytest.raises(FileExistsError):
        gen.generate_experiment(ensemble)


def test_full_exp(fileutils, test_dir, wlmutils):
    exp = Experiment("gen-test", test_dir, launcher="local")

    application = exp.create_application("application", run_settings=rs)
    script = fileutils.get_test_conf_path("sleep.py")
    application.attach_generator_files(to_copy=script)

    feature_store = FeatureStore(wlmutils.get_test_port())
    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble("test_ens", params=params, run_settings=rs)

    config = get_gen_file(fileutils, "in.atm")
    ensemble.attach_generator_files(to_configure=config)
    exp.generate(feature_store, ensemble, application)

    # test for ensemble
    assert osp.isdir(osp.join(test_dir, "test_ens/"))
    for i in range(9):
        assert osp.isdir(osp.join(test_dir, "test_ens/test_ens_" + str(i)))

    # test for feature_store dir
    assert osp.isdir(osp.join(test_dir, feature_store.name))

    # test for application file
    assert osp.isdir(osp.join(test_dir, "application"))
    assert osp.isfile(osp.join(test_dir, "application/sleep.py"))


def test_dir_files(fileutils, test_dir):
    """test the generate of applications with files that
    are directories with subdirectories and files
    """

    exp = Experiment("gen-test", test_dir, launcher="local")

    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble("dir_test", params=params, run_settings=rs)
    conf_dir = get_gen_file(fileutils, "test_dir")
    ensemble.attach_generator_files(to_configure=conf_dir)

    exp.generate(ensemble, tag="@")

    assert osp.isdir(osp.join(test_dir, "dir_test/"))
    for i in range(9):
        application_path = osp.join(test_dir, "dir_test/dir_test_" + str(i))
        assert osp.isdir(application_path)
        assert osp.isdir(osp.join(application_path, "test_dir_1"))
        assert osp.isfile(osp.join(application_path, "test.in"))


def test_print_files(fileutils, test_dir, capsys):
    """Test the stdout print of files attached to an ensemble"""

    exp = Experiment("print-attached-files-test", test_dir, launcher="local")

    ensemble = exp.create_ensemble("dir_test", replicas=1, run_settings=rs)
    ensemble.entities = []

    ensemble.print_attached_files()
    captured = capsys.readouterr()
    assert captured.out == "The ensemble is empty, no files to show.\n"

    params = {"THERMO": [10, 20], "STEPS": [20, 30]}
    ensemble = exp.create_ensemble("dir_test", params=params, run_settings=rs)
    gen_dir = get_gen_file(fileutils, "test_dir")
    symlink_dir = get_gen_file(fileutils, "to_symlink_dir")
    copy_dir = get_gen_file(fileutils, "to_copy_dir")

    ensemble.print_attached_files()
    captured = capsys.readouterr()
    expected_out = (
        tabulate(
            [
                [application.name, "No file attached to this application."]
                for application in ensemble.applications
            ],
            headers=["Application name", "Files"],
            tablefmt="grid",
        )
        + "\n"
    )

    assert captured.out == expected_out

    ensemble.attach_generator_files()
    ensemble.print_attached_files()
    captured = capsys.readouterr()
    expected_out = (
        tabulate(
            [
                [application.name, "No file attached to this entity."]
                for application in ensemble.applications
            ],
            headers=["Application name", "Files"],
            tablefmt="grid",
        )
        + "\n"
    )
    assert captured.out == expected_out

    ensemble.attach_generator_files(
        to_configure=[gen_dir, copy_dir], to_copy=copy_dir, to_symlink=symlink_dir
    )

    expected_out = tabulate(
        [
            ["Copy", copy_dir],
            ["Symlink", symlink_dir],
            ["Configure", f"{gen_dir}\n{copy_dir}"],
        ],
        headers=["Strategy", "Files"],
        tablefmt="grid",
    )

    assert all(
        str(application.files) == expected_out for application in ensemble.applications
    )

    expected_out_multi = (
        tabulate(
            [[application.name, expected_out] for application in ensemble.applications],
            headers=["Application name", "Files"],
            tablefmt="grid",
        )
        + "\n"
    )
    ensemble.print_attached_files()

    captured = capsys.readouterr()
    assert captured.out == expected_out_multi


def test_multiple_tags(fileutils, test_dir):
    """Test substitution of multiple tagged parameters on same line"""

    exp = Experiment("test-multiple-tags", test_dir)
    application_params = {"port": 6379, "password": "unbreakable_password"}
    application_settings = RunSettings("bash", "multi_tags_template.sh")
    parameterized_application = exp.create_application(
        "multi-tags", run_settings=application_settings, params=application_params
    )
    config = get_gen_file(fileutils, "multi_tags_template.sh")
    parameterized_application.attach_generator_files(to_configure=[config])
    exp.generate(parameterized_application, overwrite=True)
    exp.start(parameterized_application, block=True)

    with open(osp.join(parameterized_application.path, "multi-tags.out")) as f:
        log_content = f.read()
        assert "My two parameters are 6379 and unbreakable_password, OK?" in log_content


def test_generation_log(fileutils, test_dir):
    """Test that an error is issued when a tag is unused and make_fatal is True"""

    exp = Experiment("gen-log-test", test_dir, launcher="local")

    params = {"THERMO": [10, 20], "STEPS": [10, 20]}
    ensemble = exp.create_ensemble("dir_test", params=params, run_settings=rs)
    conf_file = get_gen_file(fileutils, "in.atm")
    ensemble.attach_generator_files(to_configure=conf_file)

    def not_header(line):
        """you can add other general checks in here"""
        return not line.startswith("Generation start date and time:")

    exp.generate(ensemble, verbose=True)

    log_file = osp.join(test_dir, "smartsim_params.txt")
    ground_truth = get_gen_file(
        fileutils, osp.join("log_params", "smartsim_params.txt")
    )

    with open(log_file) as f1, open(ground_truth) as f2:
        assert not not_header(f1.readline())
        f1 = filter(not_header, f1)
        f2 = filter(not_header, f2)
        assert all(x == y for x, y in zip(f1, f2))

    for entity in ensemble:
        assert filecmp.cmp(
            osp.join(entity.path, "smartsim_params.txt"),
            get_gen_file(
                fileutils,
                osp.join("log_params", "dir_test", entity.name, "smartsim_params.txt"),
            ),
        )


def test_config_dir(fileutils, test_dir):
    """Test the generation and configuration of applications with
    tagged files that are directories with subdirectories and files
    """
    exp = Experiment("config-dir", launcher="local")

    gen = Generator(test_dir)

    params = {"PARAM0": [0, 1], "PARAM1": [2, 3]}
    ensemble = exp.create_ensemble("test", params=params, run_settings=rs)

    config = get_gen_file(fileutils, "tag_dir_template")
    ensemble.attach_generator_files(to_configure=config)
    gen.generate_experiment(ensemble)

    assert osp.isdir(osp.join(test_dir, "test"))

    def _check_generated(test_num, param_0, param_1):
        conf_test_dir = osp.join(test_dir, "test", f"test_{test_num}")
        assert osp.isdir(conf_test_dir)
        assert osp.isdir(osp.join(conf_test_dir, "nested_0"))
        assert osp.isdir(osp.join(conf_test_dir, "nested_1"))

        with open(osp.join(conf_test_dir, "nested_0", "tagged_0.sh")) as f:
            line = f.readline()
            assert line.strip() == f'echo "Hello with parameter 0 = {param_0}"'

        with open(osp.join(conf_test_dir, "nested_1", "tagged_1.sh")) as f:
            line = f.readline()
            assert line.strip() == f'echo "Hello with parameter 1 = {param_1}"'

    _check_generated(0, 0, 2)
    _check_generated(1, 0, 3)
    _check_generated(2, 1, 2)
    _check_generated(3, 1, 3)


def test_no_gen_if_file_not_exist(fileutils):
    """Test that generation of file with non-existant config
    raises a FileNotFound exception
    """
    exp = Experiment("file-not-found", launcher="local")
    ensemble = exp.create_ensemble("test", params={"P": [0, 1]}, run_settings=rs)
    config = get_gen_file(fileutils, "path_not_exist")
    with pytest.raises(FileNotFoundError):
        ensemble.attach_generator_files(to_configure=config)


def test_no_gen_if_symlink_to_dir(fileutils):
    """Test that when configuring a directory containing a symlink
    a ValueError exception is raised to prevent circular file
    structure configuration
    """
    exp = Experiment("circular-config-files", launcher="local")
    ensemble = exp.create_ensemble("test", params={"P": [0, 1]}, run_settings=rs)
    config = get_gen_file(fileutils, "circular_config")
    with pytest.raises(ValueError):
        ensemble.attach_generator_files(to_configure=config)


def test_no_file_overwrite():
    exp = Experiment("test_no_file_overwrite", launcher="local")
    ensemble = exp.create_ensemble("test", params={"P": [0, 1]}, run_settings=rs)
    with pytest.raises(ValueError):
        ensemble.attach_generator_files(
            to_configure=["/normal/file.txt", "/path/to/smartsim_params.txt"]
        )
    with pytest.raises(ValueError):
        ensemble.attach_generator_files(
            to_symlink=["/normal/file.txt", "/path/to/smartsim_params.txt"]
        )
    with pytest.raises(ValueError):
        ensemble.attach_generator_files(
            to_copy=["/normal/file.txt", "/path/to/smartsim_params.txt"]
        )
