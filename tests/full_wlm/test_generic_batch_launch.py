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

from time import sleep

import pytest

from smartsim import Experiment, status

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_batch_model(fileutils, wlmutils):
    """Test the launch of a manually construced batch model"""

    exp_name = "test-batch-model"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir()

    script = fileutils.get_test_conf_path("sleep.py")
    batch_settings = exp.create_batch_settings(nodes=1, time="00:01:00")

    batch_settings.set_account(wlmutils.get_test_account())
    if wlmutils.get_test_launcher() == "cobalt":
        batch_settings.set_queue("debug-flat-quad")
    run_settings = wlmutils.get_run_settings("python", f"{script} --time=5")
    model = exp.create_model(
        "model", path=test_dir, run_settings=run_settings, batch_settings=batch_settings
    )
    model.set_path(test_dir)

    exp.start(model, block=True)
    statuses = exp.get_status(model)
    assert len(statuses) == 1
    assert statuses[0] == status.STATUS_COMPLETED


def test_batch_ensemble(fileutils, wlmutils):
    """Test the launch of a manually constructed batch ensemble"""

    exp_name = "test-batch-ensemble"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir()

    script = fileutils.get_test_conf_path("sleep.py")
    settings = wlmutils.get_run_settings("python", f"{script} --time=5")
    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings)

    batch = exp.create_batch_settings(nodes=1, time="00:01:00")

    batch.set_account(wlmutils.get_test_account())
    if wlmutils.get_test_launcher() == "cobalt":
        batch.set_queue("debug-flat-quad")
    ensemble = exp.create_ensemble("batch-ens", batch_settings=batch)
    ensemble.add_model(M1)
    ensemble.add_model(M2)
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def test_batch_ensemble_replicas(fileutils, wlmutils):
    exp_name = "test-batch-ensemble-replicas"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir()

    script = fileutils.get_test_conf_path("sleep.py")
    settings = wlmutils.get_run_settings("python", f"{script} --time=5")

    batch = exp.create_batch_settings(nodes=1, time="00:01:00")

    batch.set_account(wlmutils.get_test_account())
    if wlmutils.get_test_launcher() == "cobalt":
        # As Cobalt won't allow us to run two
        # jobs in the same debug queue, we need
        # to make sure the previous test's one is over
        sleep(30)
        batch.set_queue("debug-flat-quad")
    ensemble = exp.create_ensemble(
        "batch-ens-replicas", batch_settings=batch, run_settings=settings, replicas=2
    )
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])
