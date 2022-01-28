from os import path as osp

import pytest

from smartsim import Experiment
from smartsim._core.generation import Generator
from smartsim.database import Orchestrator
from smartsim.settings import RunSettings

rs = RunSettings("python", exe_args="sleep.py")


"""
Test the generation of files and input data for an experiment

TODO
 - test lists of inputs for each file type
 - test empty directories
 - test re-generation

"""


def test_ensemble(fileutils):
    exp = Experiment("gen-test", launcher="local")
    test_dir = fileutils.get_test_dir("gen_ensemble_test")
    gen = Generator(test_dir)

    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble("test", params=params, run_settings=rs)

    config = fileutils.get_test_conf_path("in.atm")
    ensemble.attach_generator_files(to_configure=config)
    gen.generate_experiment(ensemble)

    assert len(ensemble) == 9
    assert osp.isdir(osp.join(test_dir, "test"))
    for i in range(9):
        assert osp.isdir(osp.join(test_dir, "test/test_" + str(i)))


def test_ensemble_overwrite(fileutils):
    exp = Experiment("gen-test-overwrite", launcher="local")
    test_dir = fileutils.get_test_dir("test_gen_overwrite")
    gen = Generator(test_dir, overwrite=True)

    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble("test", params=params, run_settings=rs)

    config = fileutils.get_test_conf_path("in.atm")
    ensemble.attach_generator_files(to_configure=[config])
    gen.generate_experiment(ensemble)

    # re generate without overwrite
    config = fileutils.get_test_conf_path("in.atm")
    ensemble.attach_generator_files(to_configure=[config])
    gen.generate_experiment(ensemble)

    assert len(ensemble) == 9
    assert osp.isdir(osp.join(test_dir, "test"))
    for i in range(9):
        assert osp.isdir(osp.join(test_dir, "test/test_" + str(i)))


def test_ensemble_overwrite_error(fileutils):
    exp = Experiment("gen-test-overwrite-error", launcher="local")
    test_dir = fileutils.get_test_dir("test_gen_overwrite_error")
    gen = Generator(test_dir)

    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble("test", params=params, run_settings=rs)

    config = fileutils.get_test_conf_path("in.atm")
    ensemble.attach_generator_files(to_configure=[config])
    gen.generate_experiment(ensemble)

    # re generate without overwrite
    config = fileutils.get_test_conf_path("in.atm")
    ensemble.attach_generator_files(to_configure=[config])
    with pytest.raises(FileExistsError):
        gen.generate_experiment(ensemble)


def test_full_exp(fileutils):

    test_dir = fileutils.make_test_dir("gen_full_test")
    exp = Experiment("gen-test", test_dir, launcher="local")

    model = exp.create_model("model", run_settings=rs)
    script = fileutils.get_test_conf_path("sleep.py")
    model.attach_generator_files(to_copy=script)

    orc = Orchestrator(6780)
    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble("test_ens", params=params, run_settings=rs)

    config = fileutils.get_test_conf_path("in.atm")
    ensemble.attach_generator_files(to_configure=config)
    exp.generate(orc, ensemble, model)

    # test for ensemble
    assert osp.isdir(osp.join(test_dir, "test_ens/"))
    for i in range(9):
        assert osp.isdir(osp.join(test_dir, "test_ens/test_ens_" + str(i)))

    # test for orc dir
    assert osp.isdir(osp.join(test_dir, "database"))

    # test for model file
    assert osp.isdir(osp.join(test_dir, "model"))
    assert osp.isfile(osp.join(test_dir, "model/sleep.py"))


def test_dir_files(fileutils):
    """test the generate of models with files that
    are directories with subdirectories and files
    """

    test_dir = fileutils.make_test_dir("gen_dir_test")
    exp = Experiment("gen-test", test_dir, launcher="local")

    params = {"THERMO": [10, 20, 30], "STEPS": [10, 20, 30]}
    ensemble = exp.create_ensemble("dir_test", params=params, run_settings=rs)
    conf_dir = fileutils.get_test_dir_path("test_dir")
    ensemble.attach_generator_files(to_copy=conf_dir)

    exp.generate(ensemble)

    assert osp.isdir(osp.join(test_dir, "dir_test/"))
    for i in range(9):
        model_path = osp.join(test_dir, "dir_test/dir_test_" + str(i))
        assert osp.isdir(model_path)
        assert osp.isdir(osp.join(model_path, "test_dir_1"))
        assert osp.isfile(osp.join(model_path, "test.py"))


def test_multiple_tags(fileutils):
    """Test substitution of multiple tagged parameters on same line"""
    test_dir = fileutils.make_test_dir("multiple_tags")

    exp = Experiment("test-multiple-tags", test_dir)
    model_params = {"port": 6379, "password": "unbreakable_password"}
    model_settings = RunSettings("bash", "multi_tags_template.sh")
    parameterized_model = exp.create_model(
        "multi-tags", run_settings=model_settings, params=model_params
    )
    config = fileutils.get_test_conf_path("multi_tags_template.sh")
    parameterized_model.attach_generator_files(to_configure=[config])
    exp.generate(parameterized_model, overwrite=True)
    exp.start(parameterized_model, block=True)

    with open(osp.join(parameterized_model.path, "multi-tags.out")) as f:
        line = f.readline()
        assert (
            line.strip() == "My two parameters are 6379 and unbreakable_password, OK?"
        )


def test_config_dir(fileutils):
    """Test the generation and configuration of models with
    tagged files that are directories with subdirectories and files
    """
    exp = Experiment("config-dir", launcher="local")
    test_dir = fileutils.make_test_dir("test_config_dir")
    gen = Generator(test_dir)

    params = {"PARAM0": [0, 1], "PARAM1": [2, 3]}
    ensemble = exp.create_ensemble("test", params=params, run_settings=rs)

    config = fileutils.get_test_conf_path("tag_dir_template")
    ensemble.attach_generator_files(to_configure=config)
    gen.generate_experiment(ensemble)

    assert osp.isdir(osp.join(test_dir, "test"))
    # assert False
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
    config = fileutils.get_test_conf_path("path_not_exist")
    with pytest.raises(FileNotFoundError):
        ensemble.attach_generator_files(to_configure=config)


def test_no_gen_if_symlink_to_dir(fileutils):
    """Test that when configuring a directory containing a symlink
    a ValueError exception is raised to prevent circular file
    structure configuration
    """
    exp = Experiment("circular-config-files", launcher="local")
    ensemble = exp.create_ensemble("test", params={"P": [0, 1]}, run_settings=rs)
    config = fileutils.get_test_conf_path("circular_config")
    with pytest.raises(ValueError):
        ensemble.attach_generator_files(to_configure=config)
