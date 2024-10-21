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

from smartsim import Experiment
from smartsim.error import SmartSimError
from smartsim.status import JobStatus

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_failed_status(fileutils, test_dir, wlmutils):
    """Test when a failure occurs deep into application execution"""

    exp_name = "test-report-failure"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)

    script = fileutils.get_test_conf_path("bad.py")
    settings = exp.create_run_settings(
        "python", f"{script} --time=7", run_comamnd="auto"
    )

    application = exp.create_application(
        "bad-application", path=test_dir, run_settings=settings
    )

    exp.start(application, block=False)
    while not exp.finished(application):
        time.sleep(2)
    stat = exp.get_status(application)
    assert len(stat) == 1
    assert stat[0] == JobStatus.FAILED


def test_bad_run_command_args(fileutils, test_dir, wlmutils):
    """Should fail because of incorrect arguments given to the
    run command

    This test ensures that we catch immediate failures
    """
    launcher = wlmutils.get_test_launcher()
    if launcher != "slurm":
        pytest.skip(f"Only fails with slurm. Launcher is {launcher}")

    exp_name = "test-bad-run-command-args"
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    script = fileutils.get_test_conf_path("bad.py")

    # this argument will get turned into an argument for the run command
    # of the specific WLM of the system.
    settings = exp.create_run_settings(
        "python", f"{script} --time=5", run_args={"badarg": "badvalue"}
    )

    application = exp.create_application(
        "bad-application", path=test_dir, run_settings=settings
    )

    with pytest.raises(SmartSimError):
        exp.start(application)
