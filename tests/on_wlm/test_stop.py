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

import time

import pytest

from smartsim import Experiment, status

"""
Test Stopping launched entities.

These tests will have their run settings automatically created
by the experiment which will choose the run_command so runtime may vary.
"""

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_stop_entity(fileutils, test_dir, wlmutils):
    exp_name = "test-launch-stop-model"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=10")
    settings.set_tasks(1)
    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)

    exp.start(M1, block=False)
    time.sleep(5)
    exp.stop(M1)
    assert M1.name in exp._control._jobs.completed
    assert exp.get_status(M1)[0] == status.STATUS_CANCELLED


def test_stop_entity_list(fileutils, test_dir, wlmutils):
    exp_name = "test-launch-stop-ensemble"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=10")
    settings.set_tasks(1)

    ensemble = exp.create_ensemble("e1", run_settings=settings, replicas=2)
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=False)
    time.sleep(5)
    exp.stop(ensemble)
    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_CANCELLED for stat in statuses])
    assert all([m.name in exp._control._jobs.completed for m in ensemble])
