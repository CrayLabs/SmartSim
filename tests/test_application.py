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

from glob import glob
from os import path as osp

import pytest

from smartsim.entity.application import Application
from smartsim.entity.files import EntityFiles
from smartsim.settings.launch_settings import LaunchSettings

pytestmark = pytest.mark.group_a


@pytest.fixture
def get_gen_configure_dir(fileutils):
    yield fileutils.get_test_conf_path(osp.join("generator_files", "tag_dir_template"))


@pytest.fixture
def mock_launcher_settings(wlmutils):
    return LaunchSettings(wlmutils.get_test_launcher(), {}, {})


def test_application_exe_property():
    a = Application(
        "test_name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    exe = a.exe
    assert exe is a.exe


def test_application_exe_args_property():
    a = Application("test_name", exe="echo", exe_args=["spam", "eggs"])
    exe_args = a.exe_args
    assert exe_args is a.exe_args


def test_application_files_property(get_gen_configure_dir):
    tagged_files = sorted(glob(get_gen_configure_dir + "/*"))
    files = EntityFiles(tagged=tagged_files)
    a = Application("test_name", exe="echo", exe_args=["spam", "eggs"], files=files)
    files = a.files
    assert files is a.files


def test_application_file_parameters_property():
    file_parameters = {"h": [5, 6, 7, 8]}
    a = Application(
        "test_name",
        exe="echo",
        file_parameters=file_parameters,
    )
    file_parameters = a.file_parameters

    assert file_parameters is a.file_parameters


def test_application_key_prefixing_property():
    key_prefixing_enabled = True
    a = Application("test_name", exe="echo", exe_args=["spam", "eggs"])
    key_prefixing_enabled = a.key_prefixing_enabled
    assert key_prefixing_enabled == a.key_prefixing_enabled


def test_empty_executable():
    """Test that an error is raised when the exe property is empty"""
    with pytest.raises(ValueError):
        Application(name="application", exe=None, exe_args=None)


def test_executable_is_not_empty_str():
    """Test that an error is raised when the exe property is and empty str"""
    app = Application(name="application", exe="echo", exe_args=None)
    with pytest.raises(ValueError):
        app.exe = ""


def test_type_exe():
    with pytest.raises(TypeError):
        Application(
            "test_name",
            exe=2,
            exe_args=["spam", "eggs"],
        )


def test_type_exe_args():
    application = Application(
        "test_name",
        exe="echo",
    )
    with pytest.raises(TypeError):
        application.exe_args = [1, 2, 3]


def test_type_files_property():
    application = Application(
        "test_name",
        exe="echo",
    )
    with pytest.raises(TypeError):
        application.files = "/path/to/file"


def test_type_file_parameters_property():
    application = Application(
        "test_name",
        exe="echo",
    )
    with pytest.raises(TypeError):
        application.file_parameters = {1: 2}


def test_type_incoming_entities():
    application = Application(
        "test_name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(TypeError):
        application.incoming_entities = [1, 2, 3]


# application type checks
def test_application_type_exe():
    application = Application(
        "test_name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(TypeError, match="exe argument was not of type str"):
        application.exe = 2


def test_application_type_exe_args():
    application = Application(
        "test_name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        TypeError, match="Executable arguments were not a list of str or a str."
    ):
        application.exe_args = [1, 2, 3]


def test_application_type_files():
    application = Application(
        "test_name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(TypeError, match="files argument was not of type EntityFiles"):
        application.files = 2


@pytest.mark.parametrize(
    "file_params",
    (
        pytest.param(["invalid"], id="Not a mapping"),
        pytest.param({"1": 2}, id="Value is not mapping of str and str"),
        pytest.param({1: "2"}, id="Key is not mapping of str and str"),
        pytest.param({1: 2}, id="Values not mapping of str and str"),
    ),
)
def test_application_type_file_parameters(file_params):
    application = Application(
        "test_name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        TypeError,
        match="file_parameters argument was not of type mapping of str and str",
    ):
        application.file_parameters = file_params


def test_application_type_incoming_entities():

    application = Application(
        "test_name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        TypeError,
        match="incoming_entities argument was not of type list of SmartSimEntity",
    ):
        application.incoming_entities = [1, 2, 3]


def test_application_type_key_prefixing_enabled():

    application = Application(
        "test_name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        TypeError,
        match="key_prefixing_enabled argument was not of type bool",
    ):
        application.key_prefixing_enabled = "invalid"


def test_application_type_build_exe_args():
    application = Application(
        "test_name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        TypeError, match="Executable arguments were not a list of str or a str."
    ):

        application.exe_args = [1, 2, 3]
