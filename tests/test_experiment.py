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
import os

import pytest

from smartsim import Experiment
from smartsim._core.config import CONFIG
from smartsim.entity import Model
from smartsim.error import SmartSimError
from smartsim.error.errors import SSUnsupportedError
from smartsim.settings import RunSettings
from smartsim.status import STATUS_NEVER_STARTED

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.slow_tests


def test_model_prefix(test_dir):
    exp_name = "test_prefix"
    exp = Experiment(exp_name)

    model = exp.create_model(
        "model",
        path=test_dir,
        run_settings=RunSettings("python"),
        enable_key_prefixing=True,
    )
    assert model._key_prefixing_enabled == True


def test_bad_exp_path():
    with pytest.raises(NotADirectoryError):
        exp = Experiment("test", "not-a-directory")


def test_type_exp_path():
    with pytest.raises(TypeError):
        exp = Experiment("test", ["this-is-a-list-dummy"])


def test_stop_type():
    """Wrong argument type given to stop"""
    exp = Experiment("name")
    with pytest.raises(TypeError):
        exp.stop("model")


def test_finished_new_model():
    # finished should fail as this model hasn't been
    # launched yet.

    model = Model("name", {}, "./", RunSettings("python"))
    exp = Experiment("test")
    with pytest.raises(ValueError):
        exp.finished(model)


def test_status_typeerror():
    exp = Experiment("test")
    with pytest.raises(TypeError):
        exp.get_status([])


def test_status_pre_launch():
    model = Model("name", {}, "./", RunSettings("python"))
    exp = Experiment("test")
    assert exp.get_status(model)[0] == STATUS_NEVER_STARTED


def test_bad_ensemble_init_no_rs():
    """params supplied without run settings"""
    exp = Experiment("test")
    with pytest.raises(SmartSimError):
        exp.create_ensemble("name", {"param1": 1})


def test_bad_ensemble_init_no_params():
    """params supplied without run settings"""
    exp = Experiment("test")
    with pytest.raises(SmartSimError):
        exp.create_ensemble("name", run_settings=RunSettings("python"))


def test_bad_ensemble_init_no_rs_bs():
    """ensemble init without run settings or batch settings"""
    exp = Experiment("test")
    with pytest.raises(SmartSimError):
        exp.create_ensemble("name")


def test_stop_entity(test_dir):
    exp_name = "test_stop_entity"
    exp = Experiment(exp_name, exp_path=test_dir)
    m = exp.create_model("model", path=test_dir, run_settings=RunSettings("sleep", "5"))
    exp.start(m, block=False)
    assert exp.finished(m) == False
    exp.stop(m)
    assert exp.finished(m) == True


def test_poll(test_dir):
    # Ensure that a SmartSimError is not raised
    exp_name = "test_exp_poll"
    exp = Experiment(exp_name, exp_path=test_dir)
    model = exp.create_model(
        "model", path=test_dir, run_settings=RunSettings("sleep", "5")
    )
    exp.start(model, block=False)
    exp.poll(interval=1)
    exp.stop(model)


def test_summary(test_dir):
    exp_name = "test_exp_summary"
    exp = Experiment(exp_name, exp_path=test_dir)
    m = exp.create_model(
        "model", path=test_dir, run_settings=RunSettings("echo", "Hello")
    )
    exp.start(m)
    summary_str = exp.summary(style="plain")
    print(summary_str)

    summary_lines = summary_str.split("\n")
    assert 2 == len(summary_lines)

    headers, values = [s.split() for s in summary_lines]
    headers = ["Index"] + headers

    row = dict(zip(headers, values))
    assert m.name == row["Name"]
    assert m.type == row["Entity-Type"]
    assert 0 == int(row["RunID"])
    assert 0 == int(row["Returncode"])


def test_launcher_detection(wlmutils, monkeypatch):
    if wlmutils.get_test_launcher() == "pals":
        pytest.skip(reason="Launcher detection cannot currently detect pbs vs pals")
    if wlmutils.get_test_launcher() == "local":
        monkeypatch.setenv("PATH", "")  # Remove all WLMs from PATH

    exp = Experiment("test-launcher-detection", launcher="auto")

    assert exp._launcher == wlmutils.get_test_launcher()


def test_enable_disable_telemtery(monkeypatch):
    # TODO: Currently these are implemented by setting an environment variable
    #       so that ALL experiments instanced in a driver script will begin
    #       producing telemetry data. In the future it is planned to have this
    #       work on a "per-instance" basis
    monkeypatch.setattr(os, "environ", {})
    exp = Experiment("my-exp")
    exp.enable_telemetry()
    assert CONFIG.telemetry_enabled
    exp.disable_telemetry()
    assert not CONFIG.telemetry_enabled


def test_error_on_cobalt():
    with pytest.raises(SSUnsupportedError):
        exp = Experiment("cobalt_exp", launcher="cobalt")
