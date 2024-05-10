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

import os.path
from copy import deepcopy
from pathlib import Path

import pytest

from smartsim import Experiment
from smartsim.status import SmartSimStatus

"""
Test the launch of simple entity types on pre-existing allocations.

All entities will obtain the allocation from the environment of the
user

Each of the tests below will have their RunSettings automatically
created which means that these tests may vary in run command that
is used.
"""

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_models(fileutils, test_dir, wlmutils):
    exp_name = "test-models-launch"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=5")
    settings.set_tasks(1)

    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)
    M2 = exp.create_model("m2", path=test_dir, run_settings=deepcopy(settings))

    exp.start(M1, M2, block=True)
    statuses = exp.get_status(M1, M2)
    assert all([stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses])


def test_multinode_app(mpi_app_path, test_dir, wlmutils):

    if not mpi_app_path:
        pytest.skip("Test needs MPI to run")

    exp_name = "test-mpi-app"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    settings = exp.create_run_settings(str(mpi_app_path), [])
    settings.set_nodes(3)

    model = exp.create_model("mpi_app", run_settings=settings)
    exp.generate(model)

    exp.start(model, block=True)

    p = Path(model.path)
    output_files = sorted([str(path) for path in p.glob("mpi_hello*")])
    expected_files = sorted(
        [os.path.join(model.path, f"mpi_hello.{idx}.log") for idx in range(3)]
    )

    assert output_files == expected_files

    for index, file in enumerate(output_files):
        with open(file) as f:
            assert f.readlines() == [
                f"Hello world from rank {index} out of 3 processors\n"
            ]


def test_ensemble(fileutils, test_dir, wlmutils):
    exp_name = "test-ensemble-launch"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=5")
    settings.set_tasks(1)

    ensemble = exp.create_ensemble("e1", run_settings=settings, replicas=2)

    exp.start(ensemble, block=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses])


def test_summary(fileutils, test_dir, wlmutils):
    """Fairly rudimentary test of the summary dataframe"""

    exp_name = "test-launch-summary"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    sleep_exp = fileutils.get_test_conf_path("sleep.py")
    bad = fileutils.get_test_conf_path("bad.py")

    sleep_settings = exp.create_run_settings("python", f"{sleep_exp} --time=3")
    sleep_settings.set_tasks(1)
    bad_settings = exp.create_run_settings("python", f"{bad} --time=6")
    bad_settings.set_tasks(1)

    sleep_exp = exp.create_model("sleep", path=test_dir, run_settings=sleep_settings)
    bad = exp.create_model("bad", path=test_dir, run_settings=bad_settings)

    # start and poll
    exp.start(sleep_exp, bad)
    assert exp.get_status(bad)[0] == SmartSimStatus.STATUS_FAILED
    assert exp.get_status(sleep_exp)[0] == SmartSimStatus.STATUS_COMPLETED

    summary_str = exp.summary(style="plain")
    print(summary_str)

    rows = [s.split() for s in summary_str.split("\n")]
    headers = ["Index"] + rows.pop(0)

    # There is no guarantee that the order of
    # the rows will be sleep, bad
    row = dict(zip(headers, rows[0]))
    row_1 = dict(zip(headers, rows[1]))
    if row["Name"] != sleep_exp.name:
        row_1, row = row, row_1

    assert sleep_exp.name == row["Name"]
    assert sleep_exp.type == row["Entity-Type"]
    assert 0 == int(row["RunID"])
    assert 0 == int(row["Returncode"])

    assert bad.name == row_1["Name"]
    assert bad.type == row_1["Entity-Type"]
    assert 0 == int(row_1["RunID"])
    assert 0 != int(row_1["Returncode"])
