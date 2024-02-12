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
Test the launch and stop of models and ensembles using base
RunSettings while on WLM.
"""

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_model_on_wlm(fileutils, test_dir, wlmutils):
    exp_name = "test-base-settings-model-launch"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    script = fileutils.get_test_conf_path("sleep.py")
    settings1 = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    settings2 = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    M1 = exp.create_model("m1", path=test_dir, run_settings=settings1)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings2)

    # launch models twice to show that they can also be restarted
    for _ in range(2):
        exp.start(M1, M2, block=True)
        statuses = exp.get_status(M1, M2)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def test_model_stop_on_wlm(fileutils, test_dir, wlmutils):
    exp_name = "test-base-settings-model-stop"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    script = fileutils.get_test_conf_path("sleep.py")
    settings1 = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    settings2 = wlmutils.get_base_run_settings("python", f"{script} --time=5")
    M1 = exp.create_model("m1", path=test_dir, run_settings=settings1)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings2)

    # stop launched models
    exp.start(M1, M2, block=False)
    time.sleep(2)
    exp.stop(M1, M2)
    assert M1.name in exp._control._jobs.completed
    assert M2.name in exp._control._jobs.completed
    statuses = exp.get_status(M1, M2)
    assert all([stat == status.STATUS_CANCELLED for stat in statuses])
