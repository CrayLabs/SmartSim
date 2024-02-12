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


import pytest

from smartsim import Experiment, status
from smartsim.database import Orchestrator
from smartsim.error import SSUnsupportedError
from smartsim.settings import JsrunSettings, RunSettings

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_unsupported_run_settings():
    exp_name = "test-unsupported-run-settings"
    exp = Experiment(exp_name, launcher="slurm")
    bad_settings = JsrunSettings("echo", "hello")
    model = exp.create_model("bad_rs", bad_settings)

    with pytest.raises(SSUnsupportedError):
        exp.start(model)


def test_model_failure(fileutils, test_dir):
    exp_name = "test-model-failure"
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)

    script = fileutils.get_test_conf_path("bad.py")
    settings = RunSettings("python", f"{script} --time=3")

    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)

    exp.start(M1, block=True)
    statuses = exp.get_status(M1)
    assert all([stat == status.STATUS_FAILED for stat in statuses])


def test_orchestrator_relaunch(test_dir, wlmutils):
    """Test when users try to launch second orchestrator"""
    exp_name = "test-orc-on-relaunch"
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)

    orc = Orchestrator(port=wlmutils.get_test_port())
    orc.set_path(test_dir)
    orc_1 = Orchestrator(port=wlmutils.get_test_port() + 1)
    orc_1.set_path(test_dir)
    try:
        exp.start(orc)
        exp.start(orc_1)
    finally:
        exp.stop(orc)
        exp.stop(orc_1)
