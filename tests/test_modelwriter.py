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
from distutils import dir_util
from glob import glob
from os import path

import pytest

from smartsim._core.generation.modelwriter import ModelWriter
from smartsim.error.errors import ParameterWriterError, SmartSimError
from smartsim.settings import RunSettings

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


mw_run_settings = RunSettings("python", exe_args="sleep.py")


def get_gen_file(fileutils, filename):
    return fileutils.get_test_conf_path(path.join("generator_files", filename))


def test_write_easy_configs(fileutils, test_dir):
    param_dict = {
        "5": 10,  # MOM_input
        "FIRST": "SECOND",  # example_input.i
        "17": 20,  # in.airebo
        "65": "70",  # in.atm
        "placeholder": "group leftupper region",  # in.crack
        "1200": "120",  # input.nml
    }

    conf_path = get_gen_file(fileutils, "easy/marked/")
    correct_path = get_gen_file(fileutils, "easy/correct/")
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


def test_write_med_configs(fileutils, test_dir):
    param_dict = {
        "1 0 0 0": "3 0 0 0",  # in.ellipse.gayberne
        "'noleap'": "'leap'",  # input.nml
        "'0 0.25 0.5 0.75 1.0'": "'1 0.25 0.5 1.0'",  # example_input.i
        '"spherical"': '"cartesian"',  # MOM_input
        '"spoon"': '"flat"',  # MOM_input
        "3*12.0": "3*14.0",  # MOM_input
    }

    conf_path = get_gen_file(fileutils, "med/marked/")
    correct_path = get_gen_file(fileutils, "med/correct/")

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


def test_write_new_tag_configs(fileutils, test_dir):
    """sets the tag to the dollar sign"""

    param_dict = {
        "1 0 0 0": "3 0 0 0",  # in.ellipse.gayberne
        "'noleap'": "'leap'",  # input.nml
        "'0 0.25 0.5 0.75 1.0'": "'1 0.25 0.5 1.0'",  # example_input.i
        '"spherical"': '"cartesian"',  # MOM_input
        '"spoon"': '"flat"',  # MOM_input
        "3*12.0": "3*14.0",  # MOM_input
    }

    conf_path = get_gen_file(fileutils, "new-tag/marked/")
    correct_path = get_gen_file(fileutils, "new-tag/correct/")

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


def test_write_mw_error_3(fileutils, test_dir):
    param_dict = {
        "5": 10,  # MOM_input
    }

    conf_path = get_gen_file(fileutils, "easy/marked/")

    # copy confs to gen directory
    dir_util.copy_tree(conf_path, test_dir)
    assert path.isdir(test_dir)

    # init modelwriter
    writer = ModelWriter()
    with pytest.raises(SmartSimError):
        writer.configure_tagged_model_files(
            glob(test_dir + "/*"), param_dict, make_missing_tags_fatal=True
        )
