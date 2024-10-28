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

from smartsim import Experiment
from smartsim._core.launcher.dragon.dragon_launcher import DragonLauncher
from smartsim.status import JobStatus

# retrieved from pytest fixtures
if pytest.test_launcher != "dragon":
    pytestmark = pytest.mark.skip(reason="Test is only for Dragon WLM systems")


def test_dragon_global_path(global_dragon_teardown, wlmutils, test_dir, monkeypatch):
    monkeypatch.setenv("SMARTSIM_DRAGON_SERVER_PATH", test_dir)
    exp: Experiment = Experiment(
        "test_dragon_connection",
        exp_path=test_dir,
        launcher=wlmutils.get_test_launcher(),
    )
    rs = exp.create_run_settings(exe="sleep", exe_args=["1"])
    model = exp.create_application("sleep", run_settings=rs)

    exp.generate(model)
    exp.start(model, block=True)

    try:
        assert exp.get_status(model)[0] == JobStatus.COMPLETED
    finally:
        launcher: DragonLauncher = exp._control._launcher
        launcher.cleanup()


def test_dragon_exp_path(global_dragon_teardown, wlmutils, test_dir, monkeypatch):
    monkeypatch.delenv("SMARTSIM_DRAGON_SERVER_PATH", raising=False)
    monkeypatch.delenv("_SMARTSIM_DRAGON_SERVER_PATH_EXP", raising=False)
    exp: Experiment = Experiment(
        "test_dragon_connection",
        exp_path=test_dir,
        launcher=wlmutils.get_test_launcher(),
    )
    rs = exp.create_run_settings(exe="sleep", exe_args=["1"])
    model = exp.create_application("sleep", run_settings=rs)

    exp.generate(model)
    exp.start(model, block=True)
    try:
        assert exp.get_status(model)[0] == JobStatus.COMPLETED
    finally:
        launcher: DragonLauncher = exp._control._launcher
        launcher.cleanup()


def test_dragon_cannot_honor(wlmutils, test_dir):
    exp: Experiment = Experiment(
        "test_dragon_cannot_honor",
        exp_path=test_dir,
        launcher=wlmutils.get_test_launcher(),
    )
    rs = exp.create_run_settings(exe="sleep", exe_args=["1"])
    rs.set_nodes(100)
    model = exp.create_application("sleep", run_settings=rs)

    exp.generate(model)
    exp.start(model, block=True)

    try:
        assert exp.get_status(model)[0] == JobStatus.FAILED
    finally:
        launcher: DragonLauncher = exp._control._launcher
        launcher.cleanup()
