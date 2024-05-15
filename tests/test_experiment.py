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
import os.path as osp
import pathlib
import shutil
import typing as t

import pytest

from smartsim import Experiment
from smartsim._core.config import CONFIG
from smartsim._core.config.config import Config
from smartsim._core.utils import serialize
from smartsim.database import Orchestrator
from smartsim.entity import Model
from smartsim.error import SmartSimError
from smartsim.error.errors import SSUnsupportedError
from smartsim.settings import RunSettings
from smartsim.status import SmartSimStatus

if t.TYPE_CHECKING:
    import conftest


# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.slow_tests


def test_model_prefix(test_dir: str) -> None:
    exp_name = "test_prefix"
    exp = Experiment(exp_name)

    model = exp.create_model(
        "model",
        path=test_dir,
        run_settings=RunSettings("python"),
        enable_key_prefixing=True,
    )
    assert model._key_prefixing_enabled == True


def test_model_no_name():
    exp = Experiment("test_model_no_name")
    with pytest.raises(AttributeError):
        _ = exp.create_model(name=None, run_settings=RunSettings("python"))


def test_ensemble_no_name():
    exp = Experiment("test_ensemble_no_name")
    with pytest.raises(AttributeError):
        _ = exp.create_ensemble(
            name=None, run_settings=RunSettings("python"), replicas=2
        )


def test_bad_exp_path() -> None:
    with pytest.raises(NotADirectoryError):
        exp = Experiment("test", "not-a-directory")


def test_type_exp_path() -> None:
    with pytest.raises(TypeError):
        exp = Experiment("test", ["this-is-a-list-dummy"])


def test_stop_type() -> None:
    """Wrong argument type given to stop"""
    exp = Experiment("name")
    with pytest.raises(TypeError):
        exp.stop("model")


def test_finished_new_model() -> None:
    # finished should fail as this model hasn't been
    # launched yet.

    model = Model("name", {}, "./", RunSettings("python"))
    exp = Experiment("test")
    with pytest.raises(ValueError):
        exp.finished(model)


def test_status_typeerror() -> None:
    exp = Experiment("test")
    with pytest.raises(TypeError):
        exp.get_status([])


def test_status_pre_launch() -> None:
    model = Model("name", {}, "./", RunSettings("python"))
    exp = Experiment("test")
    assert exp.get_status(model)[0] == SmartSimStatus.STATUS_NEVER_STARTED


def test_bad_ensemble_init_no_rs(test_dir: str) -> None:
    """params supplied without run settings"""
    exp = Experiment("test", exp_path=test_dir)
    with pytest.raises(SmartSimError):
        exp.create_ensemble("name", {"param1": 1})


def test_bad_ensemble_init_no_params(test_dir: str) -> None:
    """params supplied without run settings"""
    exp = Experiment("test", exp_path=test_dir)
    with pytest.raises(SmartSimError):
        exp.create_ensemble("name", run_settings=RunSettings("python"))


def test_bad_ensemble_init_no_rs_bs(test_dir: str) -> None:
    """ensemble init without run settings or batch settings"""
    exp = Experiment("test", exp_path=test_dir)
    with pytest.raises(SmartSimError):
        exp.create_ensemble("name")


def test_stop_entity(test_dir: str) -> None:
    exp_name = "test_stop_entity"
    exp = Experiment(exp_name, exp_path=test_dir)
    m = exp.create_model("model", path=test_dir, run_settings=RunSettings("sleep", "5"))
    exp.start(m, block=False)
    assert exp.finished(m) == False
    exp.stop(m)
    assert exp.finished(m) == True


def test_poll(test_dir: str) -> None:
    # Ensure that a SmartSimError is not raised
    exp_name = "test_exp_poll"
    exp = Experiment(exp_name, exp_path=test_dir)
    model = exp.create_model(
        "model", path=test_dir, run_settings=RunSettings("sleep", "5")
    )
    exp.start(model, block=False)
    exp.poll(interval=1)
    exp.stop(model)


def test_summary(test_dir: str) -> None:
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


def test_launcher_detection(
    wlmutils: "conftest.WLMUtils", monkeypatch: pytest.MonkeyPatch
) -> None:
    if wlmutils.get_test_launcher() == "pals":
        pytest.skip(reason="Launcher detection cannot currently detect pbs vs pals")
    if wlmutils.get_test_launcher() == "local":
        monkeypatch.setenv("PATH", "")  # Remove all WLMs from PATH
    if wlmutils.get_test_launcher() == "dragon":
        pytest.skip(reason="Launcher detection cannot currently detect dragon")

    exp = Experiment("test-launcher-detection", launcher="auto")

    assert exp._launcher == wlmutils.get_test_launcher()


def test_enable_disable_telemetry(
    monkeypatch: pytest.MonkeyPatch, test_dir: str, config: Config
) -> None:
    # Global telemetry defaults to `on` and can be modified by
    # setting the value of env var SMARTSIM_FLAG_TELEMETRY to 0/1
    monkeypatch.setattr(os, "environ", {})
    exp = Experiment("my-exp", exp_path=test_dir)
    exp.telemetry.enable()
    assert exp.telemetry.is_enabled

    exp.telemetry.disable()
    assert not exp.telemetry.is_enabled

    exp.telemetry.enable()
    assert exp.telemetry.is_enabled

    exp.telemetry.disable()
    assert not exp.telemetry.is_enabled

    exp.start()
    mani_path = (
        pathlib.Path(test_dir) / config.telemetry_subdir / serialize.MANIFEST_FILENAME
    )
    assert mani_path.exists()


def test_telemetry_default(
    monkeypatch: pytest.MonkeyPatch, test_dir: str, config: Config
) -> None:
    """Ensure the default values for telemetry configuration match expectation
    that experiment telemetry is on"""

    # If env var related to telemetry doesn't exist, experiment should default to True
    monkeypatch.setattr(os, "environ", {})
    exp = Experiment("my-exp", exp_path=test_dir)
    assert exp.telemetry.is_enabled

    # If telemetry disabled in env, should get False
    monkeypatch.setenv("SMARTSIM_FLAG_TELEMETRY", "0")
    exp = Experiment("my-exp", exp_path=test_dir)
    assert not exp.telemetry.is_enabled

    # If telemetry enabled in env, should get True
    monkeypatch.setenv("SMARTSIM_FLAG_TELEMETRY", "1")
    exp = Experiment("my-exp", exp_path=test_dir)
    assert exp.telemetry.is_enabled


def test_error_on_cobalt() -> None:
    with pytest.raises(SSUnsupportedError):
        exp = Experiment("cobalt_exp", launcher="cobalt")


def test_default_orch_path(
    monkeypatch: pytest.MonkeyPatch, test_dir: str, wlmutils: "conftest.WLMUtils"
) -> None:
    """Ensure the default file structure is created for Orchestrator"""

    exp_name = "default-orch-path"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)
    monkeypatch.setattr(exp._control, "start", lambda *a, **kw: ...)
    db = exp.create_database(
        port=wlmutils.get_test_port(), interface=wlmutils.get_test_interface()
    )
    exp.start(db)
    orch_path = pathlib.Path(test_dir) / db.name
    assert orch_path.exists()
    assert db.path == str(orch_path)


def test_default_model_path(
    monkeypatch: pytest.MonkeyPatch, test_dir: str, wlmutils: "conftest.WLMUtils"
) -> None:
    """Ensure the default file structure is created for Model"""

    exp_name = "default-model-path"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)
    monkeypatch.setattr(exp._control, "start", lambda *a, **kw: ...)
    settings = exp.create_run_settings(exe="echo", exe_args="hello")
    model = exp.create_model(name="model_name", run_settings=settings)
    exp.start(model)
    model_path = pathlib.Path(test_dir) / model.name
    assert model_path.exists()
    assert model.path == str(model_path)


def test_default_ensemble_path(
    monkeypatch: pytest.MonkeyPatch, test_dir: str, wlmutils: "conftest.WLMUtils"
) -> None:
    """Ensure the default file structure is created for Ensemble"""

    exp_name = "default-ensemble-path"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)
    monkeypatch.setattr(exp._control, "start", lambda *a, **kw: ...)
    settings = exp.create_run_settings(exe="echo", exe_args="hello")
    ensemble = exp.create_ensemble(
        name="ensemble_name", run_settings=settings, replicas=2
    )
    exp.start(ensemble)
    ensemble_path = pathlib.Path(test_dir) / ensemble.name
    assert ensemble_path.exists()
    assert ensemble.path == str(ensemble_path)
    for member in ensemble.models:
        member_path = ensemble_path / member.name
        assert member_path.exists()
        assert member.path == str(ensemble_path / member.name)


def test_user_orch_path(
    monkeypatch: pytest.MonkeyPatch, test_dir: str, wlmutils: "conftest.WLMUtils"
) -> None:
    """Ensure a relative path is used to created Orchestrator folder"""

    exp_name = "default-orch-path"
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)
    monkeypatch.setattr(exp._control, "start", lambda *a, **kw: ...)
    db = exp.create_database(
        port=wlmutils.get_test_port(),
        interface=wlmutils.get_test_interface(),
        path="./testing_folder1234",
    )
    exp.start(db)
    orch_path = pathlib.Path(osp.abspath("./testing_folder1234"))
    assert orch_path.exists()
    assert db.path == str(orch_path)
    shutil.rmtree(orch_path)


def test_default_model_with_path(
    monkeypatch: pytest.MonkeyPatch, test_dir: str, wlmutils: "conftest.WLMUtils"
) -> None:
    """Ensure a relative path is used to created Model folder"""

    exp_name = "default-ensemble-path"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)
    monkeypatch.setattr(exp._control, "start", lambda *a, **kw: ...)
    settings = exp.create_run_settings(exe="echo", exe_args="hello")
    model = exp.create_model(
        name="model_name", run_settings=settings, path="./testing_folder1234"
    )
    exp.start(model)
    model_path = pathlib.Path(osp.abspath("./testing_folder1234"))
    assert model_path.exists()
    assert model.path == str(model_path)
    shutil.rmtree(model_path)


def test_default_ensemble_with_path(
    monkeypatch: pytest.MonkeyPatch, test_dir: str, wlmutils: "conftest.WLMUtils"
) -> None:
    """Ensure a relative path is used to created Ensemble folder"""

    exp_name = "default-ensemble-path"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher(), exp_path=test_dir)
    monkeypatch.setattr(exp._control, "start", lambda *a, **kw: ...)
    settings = exp.create_run_settings(exe="echo", exe_args="hello")
    ensemble = exp.create_ensemble(
        name="ensemble_name",
        run_settings=settings,
        path="./testing_folder1234",
        replicas=2,
    )
    exp.start(ensemble)
    ensemble_path = pathlib.Path(osp.abspath("./testing_folder1234"))
    assert ensemble_path.exists()
    assert ensemble.path == str(ensemble_path)
    for member in ensemble.models:
        member_path = ensemble_path / member.name
        assert member_path.exists()
        assert member.path == str(member_path)
    shutil.rmtree(ensemble_path)
