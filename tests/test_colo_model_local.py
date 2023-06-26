# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

from smartsim import Experiment, status

if sys.platform == "darwin":
    supported_dbs = ["tcp", "deprecated"]
else:
    supported_dbs = ["uds", "tcp", "deprecated"]
@pytest.mark.parametrize("db_type", supported_dbs)
def test_launch_colocated_model_defaults(fileutils, coloutils, db_type, launcher="local"):
    """Test the launch of a model with a colocated database and local launcher"""

    db_args = { }

    exp = Experiment("colocated_model_defaults", launcher=launcher)
    colo_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        db_args,
    )

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

    # test restarting the colocated model
    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

@pytest.mark.parametrize("db_type", supported_dbs)
def test_colocated_model_pinning_auto_1cpu(fileutils, coloutils, db_type, launcher="local"):

    exp = Experiment("colocated_model_pinning_auto_1cpu", launcher=launcher)
    db_args = {
        "limit_db_cpus": True,
        "db_cpus": 1,
    }

    # Check to make sure that the CPU mask was correctly generated
    colo_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        db_args,
    )
    assert colo_model.run_settings.colocated_db_settings["db_cpu_list"] == "0"
    assert colo_model.run_settings.colocated_db_settings["limit_db_cpus"]
    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

@pytest.mark.parametrize("db_type", supported_dbs)
def test_colocated_model_pinning_auto_2cpu(fileutils, coloutils, db_type, launcher="local"):

    exp = Experiment("colocated_model_pinning_auto_2cpu", launcher=launcher)

    db_args = {
        "limit_db_cpus": True,
        "db_cpus": 2,
    }

    # Check to make sure that the CPU mask was correctly generated
    colo_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        db_args,
    )
    assert colo_model.run_settings.colocated_db_settings["db_cpu_list"] == "0-1"
    assert colo_model.run_settings.colocated_db_settings["limit_db_cpus"]
    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

@pytest.mark.parametrize("db_type", supported_dbs)
def test_colocated_model_pinning_manual(fileutils, coloutils, db_type, launcher="local"):
    # Check to make sure that the CPU mask was correctly generated

    exp = Experiment("colocated_model_pinning_manual", launcher=launcher)

    db_args = {
        "limit_db_cpus": True,
        "db_cpus": 2,
        "db_cpu_list": "0,2"
    }

    colo_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        db_args,
    )
    assert colo_model.run_settings.colocated_db_settings["db_cpu_list"] == "0,2"
    assert colo_model.run_settings.colocated_db_settings["limit_db_cpus"]
    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])