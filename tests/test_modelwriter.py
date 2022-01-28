import filecmp
from distutils import dir_util
from glob import glob
from os import path

import pytest

from smartsim._core.generation.modelwriter import ModelWriter
from smartsim.error.errors import ParameterWriterError
from smartsim.settings import RunSettings

mw_run_settings = RunSettings("python", exe_args="sleep.py")


def test_write_easy_configs(fileutils):

    test_dir = fileutils.make_test_dir("easy_modelwriter_test")

    param_dict = {
        "5": 10,  # MOM_input
        "FIRST": "SECOND",  # example_input.i
        "17": 20,  # in.airebo
        "65": "70",  # in.atm
        "placeholder": "group leftupper region",  # in.crack
        "1200": "120",  # input.nml
    }

    conf_path = fileutils.get_test_dir_path("easy/marked/")
    correct_path = fileutils.get_test_dir_path("easy/correct/")
    # copy confs to gen directory
    dir_util.copy_tree(conf_path, test_dir)
    assert path.isdir(test_dir)

    # init modelwriter
    writer = ModelWriter()
    writer.configure_tagged_model_files(glob(test_dir + "/*"), param_dict)

    written_files = sorted(glob(test_dir + "/*"))
    correct_files = sorted(glob(correct_path + "*"))

    for written, correct in zip(written_files, correct_files):
        assert filecmp.cmp(written, correct)


def test_write_med_configs(fileutils):

    test_dir = fileutils.make_test_dir("med_modelwriter_test")

    param_dict = {
        "1 0 0 0": "3 0 0 0",  # in.ellipse.gayberne
        "'noleap'": "'leap'",  # input.nml
        "'0 0.25 0.5 0.75 1.0'": "'1 0.25 0.5 1.0'",  # example_input.i
        '"spherical"': '"cartesian"',  # MOM_input
        '"spoon"': '"flat"',  # MOM_input
        "3*12.0": "3*14.0",  # MOM_input
    }

    conf_path = fileutils.get_test_dir_path("med/marked/")
    correct_path = fileutils.get_test_dir_path("med/correct/")

    # copy confs to gen directory
    dir_util.copy_tree(conf_path, test_dir)
    assert path.isdir(test_dir)

    # init modelwriter
    writer = ModelWriter()
    writer.set_tag(writer.tag, "(;.+;)")
    assert writer.regex == "(;.+;)"
    writer.configure_tagged_model_files(glob(test_dir + "/*"), param_dict)

    written_files = sorted(glob(test_dir + "/*"))
    correct_files = sorted(glob(correct_path + "*"))

    for written, correct in zip(written_files, correct_files):
        assert filecmp.cmp(written, correct)


def test_write_new_tag_configs(fileutils):
    """sets the tag to the dollar sign"""

    test_dir = fileutils.make_test_dir("new_tag_modelwriter_test")

    param_dict = {
        "1 0 0 0": "3 0 0 0",  # in.ellipse.gayberne
        "'noleap'": "'leap'",  # input.nml
        "'0 0.25 0.5 0.75 1.0'": "'1 0.25 0.5 1.0'",  # example_input.i
        '"spherical"': '"cartesian"',  # MOM_input
        '"spoon"': '"flat"',  # MOM_input
        "3*12.0": "3*14.0",  # MOM_input
    }

    conf_path = fileutils.get_test_dir_path("new-tag/marked/")
    correct_path = fileutils.get_test_dir_path("new-tag/correct/")

    # copy confs to gen directory
    dir_util.copy_tree(conf_path, test_dir)
    assert path.isdir(test_dir)

    # init modelwriter
    writer = ModelWriter()
    writer.set_tag("@")
    writer.configure_tagged_model_files(glob(test_dir + "/*"), param_dict)

    written_files = sorted(glob(test_dir + "/*"))
    correct_files = sorted(glob(correct_path + "*"))

    for written, correct in zip(written_files, correct_files):
        assert filecmp.cmp(written, correct)


def test_mw_error_1():
    writer = ModelWriter()
    with pytest.raises(ParameterWriterError):
        writer.configure_tagged_model_files("[not/a/path]", {"5": 10})


def test_mw_error_2():
    writer = ModelWriter()
    with pytest.raises(ParameterWriterError):
        writer._write_changes("[not/a/path]")
