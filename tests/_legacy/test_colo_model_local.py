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

import sys

import pytest

from smartsim import Experiment
from smartsim.entity import Application
from smartsim.error import SSUnsupportedError
from smartsim.status import JobStatus

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.slow_tests


if sys.platform == "darwin":
    supported_fss = ["tcp", "deprecated"]
else:
    supported_fss = ["uds", "tcp", "deprecated"]

is_mac = sys.platform == "darwin"


@pytest.mark.skipif(not is_mac, reason="MacOS-only test")
def test_macosx_warning(fileutils, test_dir, coloutils):
    fs_args = {"custom_pinning": [1]}
    fs_type = "uds"  # Test is insensitive to choice of fs

    exp = Experiment(
        "colocated_application_defaults", launcher="local", exp_path=test_dir
    )
    with pytest.warns(
        RuntimeWarning,
        match="CPU pinning is not supported on MacOSX. Ignoring pinning specification.",
    ):
        _ = coloutils.setup_test_colo(
            fileutils,
            fs_type,
            exp,
            "send_data_local_smartredis.py",
            fs_args,
        )


def test_unsupported_limit_app(fileutils, test_dir, coloutils):
    fs_args = {"limit_app_cpus": True}
    fs_type = "uds"  # Test is insensitive to choice of fs

    exp = Experiment(
        "colocated_application_defaults", launcher="local", exp_path=test_dir
    )
    with pytest.raises(SSUnsupportedError):
        coloutils.setup_test_colo(
            fileutils,
            fs_type,
            exp,
            "send_data_local_smartredis.py",
            fs_args,
        )


@pytest.mark.skipif(is_mac, reason="Unsupported on MacOSX")
@pytest.mark.parametrize("custom_pinning", [1, "10", "#", 1.0, ["a"], [1.0]])
def test_unsupported_custom_pinning(fileutils, test_dir, coloutils, custom_pinning):
    fs_type = "uds"  # Test is insensitive to choice of fs
    fs_args = {"custom_pinning": custom_pinning}

    exp = Experiment(
        "colocated_application_defaults", launcher="local", exp_path=test_dir
    )
    with pytest.raises(TypeError):
        coloutils.setup_test_colo(
            fileutils,
            fs_type,
            exp,
            "send_data_local_smartredis.py",
            fs_args,
        )


@pytest.mark.skipif(is_mac, reason="Unsupported on MacOSX")
@pytest.mark.parametrize(
    "pin_list, num_cpus, expected",
    [
        pytest.param(None, 2, "0,1", id="Automatic creation of pinned cpu list"),
        pytest.param([1, 2], 2, "1,2", id="Individual ids only"),
        pytest.param([range(2), 3], 3, "0,1,3", id="Mixed ranges and individual ids"),
        pytest.param(range(3), 3, "0,1,2", id="Range only"),
        pytest.param(
            [range(8, 10), range(6, 1, -2)], 4, "2,4,6,8,9", id="Multiple ranges"
        ),
    ],
)
def test_create_pinning_string(pin_list, num_cpus, expected):
    assert Application._create_pinning_string(pin_list, num_cpus) == expected


@pytest.mark.parametrize("fs_type", supported_fss)
def test_launch_colocated_application_defaults(
    fileutils, test_dir, coloutils, fs_type, launcher="local"
):
    """Test the launch of a application with a colocated feature store and local launcher"""

    fs_args = {}

    exp = Experiment(
        "colocated_application_defaults", launcher=launcher, exp_path=test_dir
    )
    colo_application = coloutils.setup_test_colo(
        fileutils,
        fs_type,
        exp,
        "send_data_local_smartredis.py",
        fs_args,
    )

    if is_mac:
        true_pinning = None
    else:
        true_pinning = "0"
    assert (
        colo_application.run_settings.colocated_fs_settings["custom_pinning"]
        == true_pinning
    )
    exp.generate(colo_application)
    exp.start(colo_application, block=True)
    statuses = exp.get_status(colo_application)
    assert all(stat == JobStatus.COMPLETED for stat in statuses)

    # test restarting the colocated application
    exp.start(colo_application, block=True)
    statuses = exp.get_status(colo_application)
    assert all(stat == JobStatus.COMPLETED for stat in statuses), f"Statuses {statuses}"


@pytest.mark.parametrize("fs_type", supported_fss)
def test_launch_multiple_colocated_applications(
    fileutils, test_dir, coloutils, wlmutils, fs_type, launcher="local"
):
    """Test the concurrent launch of two applications with a colocated feature store and local launcher"""

    fs_args = {}

    exp = Experiment("multi_colo_applications", launcher=launcher, exp_path=test_dir)
    colo_applications = [
        coloutils.setup_test_colo(
            fileutils,
            fs_type,
            exp,
            "send_data_local_smartredis.py",
            fs_args,
            colo_application_name="colo0",
            port=wlmutils.get_test_port(),
        ),
        coloutils.setup_test_colo(
            fileutils,
            fs_type,
            exp,
            "send_data_local_smartredis.py",
            fs_args,
            colo_application_name="colo1",
            port=wlmutils.get_test_port() + 1,
        ),
    ]
    exp.generate(*colo_applications)
    exp.start(*colo_applications, block=True)
    statuses = exp.get_status(*colo_applications)
    assert all(stat == JobStatus.COMPLETED for stat in statuses)

    # test restarting the colocated application
    exp.start(*colo_applications, block=True)
    statuses = exp.get_status(*colo_applications)
    assert all([stat == JobStatus.COMPLETED for stat in statuses])


@pytest.mark.parametrize("fs_type", supported_fss)
def test_colocated_application_disable_pinning(
    fileutils, test_dir, coloutils, fs_type, launcher="local"
):
    exp = Experiment(
        "colocated_application_pinning_auto_1cpu", launcher=launcher, exp_path=test_dir
    )
    fs_args = {
        "fs_cpus": 1,
        "custom_pinning": [],
    }
    # Check to make sure that the CPU mask was correctly generated
    colo_application = coloutils.setup_test_colo(
        fileutils,
        fs_type,
        exp,
        "send_data_local_smartredis.py",
        fs_args,
    )
    assert colo_application.run_settings.colocated_fs_settings["custom_pinning"] is None
    exp.generate(colo_application)
    exp.start(colo_application, block=True)
    statuses = exp.get_status(colo_application)
    assert all([stat == JobStatus.COMPLETED for stat in statuses])


@pytest.mark.parametrize("fs_type", supported_fss)
def test_colocated_application_pinning_auto_2cpu(
    fileutils, test_dir, coloutils, fs_type, launcher="local"
):
    exp = Experiment(
        "colocated_application_pinning_auto_2cpu", launcher=launcher, exp_path=test_dir
    )

    fs_args = {
        "fs_cpus": 2,
    }

    # Check to make sure that the CPU mask was correctly generated
    colo_application = coloutils.setup_test_colo(
        fileutils,
        fs_type,
        exp,
        "send_data_local_smartredis.py",
        fs_args,
    )
    if is_mac:
        true_pinning = None
    else:
        true_pinning = "0,1"
    assert (
        colo_application.run_settings.colocated_fs_settings["custom_pinning"]
        == true_pinning
    )
    exp.generate(colo_application)
    exp.start(colo_application, block=True)
    statuses = exp.get_status(colo_application)
    assert all([stat == JobStatus.COMPLETED for stat in statuses])


@pytest.mark.skipif(is_mac, reason="unsupported on MacOSX")
@pytest.mark.parametrize("fs_type", supported_fss)
def test_colocated_application_pinning_range(
    fileutils, test_dir, coloutils, fs_type, launcher="local"
):
    # Check to make sure that the CPU mask was correctly generated

    exp = Experiment(
        "colocated_application_pinning_manual", launcher=launcher, exp_path=test_dir
    )

    fs_args = {"fs_cpus": 2, "custom_pinning": range(2)}

    colo_application = coloutils.setup_test_colo(
        fileutils,
        fs_type,
        exp,
        "send_data_local_smartredis.py",
        fs_args,
    )
    assert (
        colo_application.run_settings.colocated_fs_settings["custom_pinning"] == "0,1"
    )
    exp.generate(colo_application)
    exp.start(colo_application, block=True)
    statuses = exp.get_status(colo_application)
    assert all([stat == JobStatus.COMPLETED for stat in statuses])


@pytest.mark.skipif(is_mac, reason="unsupported on MacOSX")
@pytest.mark.parametrize("fs_type", supported_fss)
def test_colocated_application_pinning_list(
    fileutils, test_dir, coloutils, fs_type, launcher="local"
):
    # Check to make sure that the CPU mask was correctly generated

    exp = Experiment(
        "colocated_application_pinning_manual", launcher=launcher, exp_path=test_dir
    )

    fs_args = {"fs_cpus": 1, "custom_pinning": [1]}

    colo_application = coloutils.setup_test_colo(
        fileutils,
        fs_type,
        exp,
        "send_data_local_smartredis.py",
        fs_args,
    )
    assert colo_application.run_settings.colocated_fs_settings["custom_pinning"] == "1"
    exp.generate(colo_application)
    exp.start(colo_application, block=True)
    statuses = exp.get_status(colo_application)
    assert all([stat == JobStatus.COMPLETED for stat in statuses])


def test_colo_uds_verifies_socket_file_name(test_dir, launcher="local"):
    exp = Experiment(f"colo_uds_wrong_name", launcher=launcher, exp_path=test_dir)

    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=["--version"])

    colo_application = exp.create_application("wrong_uds_socket_name", colo_settings)

    with pytest.raises(ValueError):
        colo_application.colocate_fs_uds(unix_socket="this is not a valid name!")
