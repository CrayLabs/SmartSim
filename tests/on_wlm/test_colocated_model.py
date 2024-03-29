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
from smartsim.entity import Model
from smartsim.status import SmartSimStatus

if sys.platform == "darwin":
    supported_dbs = ["tcp", "deprecated"]
else:
    supported_dbs = ["uds", "tcp", "deprecated"]

# Set to true if DB logs should be generated for debugging
DEBUG_DB = False

# retrieved from pytest fixtures
launcher = pytest.test_launcher
if launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


@pytest.mark.parametrize("db_type", supported_dbs)
def test_launch_colocated_model_defaults(fileutils, test_dir, coloutils, db_type):
    """Test the launch of a model with a colocated database and local launcher"""

    db_args = {"debug": DEBUG_DB}

    exp = Experiment("colocated_model_defaults", launcher=launcher, exp_path=test_dir)
    colo_model = coloutils.setup_test_colo(
        fileutils, db_type, exp, "send_data_local_smartredis.py", db_args, on_wlm=True
    )
    exp.generate(colo_model)
    assert colo_model.run_settings.colocated_db_settings["custom_pinning"] == "0"
    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all(
        stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses
    ), f"Statuses: {statuses}"

    # test restarting the colocated model
    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all(
        stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses
    ), f"Statuses: {statuses}"


@pytest.mark.parametrize("db_type", supported_dbs)
def test_colocated_model_disable_pinning(fileutils, test_dir, coloutils, db_type):
    exp = Experiment(
        "colocated_model_pinning_auto_1cpu", launcher=launcher, exp_path=test_dir
    )
    db_args = {
        "db_cpus": 1,
        "custom_pinning": [],
        "debug": DEBUG_DB,
    }

    # Check to make sure that the CPU mask was correctly generated
    colo_model = coloutils.setup_test_colo(
        fileutils, db_type, exp, "send_data_local_smartredis.py", db_args, on_wlm=True
    )
    assert colo_model.run_settings.colocated_db_settings["custom_pinning"] is None
    exp.generate(colo_model)
    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all(
        stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses
    ), f"Statuses: {statuses}"


@pytest.mark.parametrize("db_type", supported_dbs)
def test_colocated_model_pinning_auto_2cpu(fileutils, test_dir, coloutils, db_type):
    exp = Experiment(
        "colocated_model_pinning_auto_2cpu",
        launcher=launcher,
        exp_path=test_dir,
    )

    db_args = {"db_cpus": 2, "debug": DEBUG_DB}

    # Check to make sure that the CPU mask was correctly generated
    colo_model = coloutils.setup_test_colo(
        fileutils, db_type, exp, "send_data_local_smartredis.py", db_args, on_wlm=True
    )
    assert colo_model.run_settings.colocated_db_settings["custom_pinning"] == "0,1"
    exp.generate(colo_model)
    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all(
        stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses
    ), f"Statuses: {statuses}"


@pytest.mark.parametrize("db_type", supported_dbs)
def test_colocated_model_pinning_range(fileutils, test_dir, coloutils, db_type):
    # Check to make sure that the CPU mask was correctly generated
    # Assume that there are at least 4 cpus on the node

    exp = Experiment(
        "colocated_model_pinning_manual",
        launcher=launcher,
        exp_path=test_dir,
    )

    db_args = {"db_cpus": 4, "custom_pinning": range(4), "debug": DEBUG_DB}

    colo_model = coloutils.setup_test_colo(
        fileutils, db_type, exp, "send_data_local_smartredis.py", db_args, on_wlm=True
    )
    assert colo_model.run_settings.colocated_db_settings["custom_pinning"] == "0,1,2,3"
    exp.generate(colo_model)
    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all(
        stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses
    ), f"Statuses: {statuses}"


@pytest.mark.parametrize("db_type", supported_dbs)
def test_colocated_model_pinning_list(fileutils, test_dir, coloutils, db_type):
    # Check to make sure that the CPU mask was correctly generated
    # note we presume that this has more than 2 CPUs on the supercomputer node

    exp = Experiment(
        "colocated_model_pinning_manual",
        launcher=launcher,
        exp_path=test_dir,
    )

    db_args = {"db_cpus": 2, "custom_pinning": [0, 2]}

    colo_model = coloutils.setup_test_colo(
        fileutils, db_type, exp, "send_data_local_smartredis.py", db_args, on_wlm=True
    )
    assert colo_model.run_settings.colocated_db_settings["custom_pinning"] == "0,2"
    exp.generate(colo_model)
    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all(
        stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses
    ), f"Statuses: {statuses}"


@pytest.mark.parametrize("db_type", supported_dbs)
def test_colocated_model_pinning_mixed(fileutils, test_dir, coloutils, db_type):
    # Check to make sure that the CPU mask was correctly generated
    # note we presume that this at least 4 CPUs on the supercomputer node

    exp = Experiment(
        "colocated_model_pinning_manual",
        launcher=launcher,
        exp_path=test_dir,
    )

    db_args = {"db_cpus": 2, "custom_pinning": [range(2), 3]}

    colo_model = coloutils.setup_test_colo(
        fileutils, db_type, exp, "send_data_local_smartredis.py", db_args, on_wlm=True
    )
    assert colo_model.run_settings.colocated_db_settings["custom_pinning"] == "0,1,3"
    exp.generate(colo_model)
    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all(
        stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses
    ), f"Statuses: {statuses}"
