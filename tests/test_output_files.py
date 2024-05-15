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
import pathlib

import pytest

from smartsim import Experiment
from smartsim._core.config import CONFIG
from smartsim._core.control.controller import Controller, _AnonymousBatchJob
from smartsim._core.launcher.step import Step
from smartsim.database.orchestrator import Orchestrator
from smartsim.entity.ensemble import Ensemble
from smartsim.entity.model import Model
from smartsim.settings.base import RunSettings
from smartsim.settings.slurmSettings import SbatchSettings, SrunSettings

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a

controller = Controller()
slurm_controller = Controller(launcher="slurm")

rs = RunSettings("echo", ["spam", "eggs"])
bs = SbatchSettings()
batch_rs = SrunSettings("echo", ["spam", "eggs"])

ens = Ensemble("ens", params={}, run_settings=rs, batch_settings=bs, replicas=3)
orc = Orchestrator(db_nodes=3, batch=True, launcher="slurm", run_command="srun")
model = Model("test_model", params={}, path="", run_settings=rs)
batch_model = Model(
    "batch_test_model", params={}, path="", run_settings=batch_rs, batch_settings=bs
)
anon_batch_model = _AnonymousBatchJob(batch_model)


def test_mutated_model_output(test_dir):
    exp_name = "test-mutated-model-output"
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)

    test_model = exp.create_model("test_model", path=test_dir, run_settings=rs)
    exp.generate(test_model)
    exp.start(test_model, block=True)

    assert pathlib.Path(test_model.path).exists()
    assert pathlib.Path(test_model.path, f"{test_model.name}.out").is_symlink()
    assert pathlib.Path(test_model.path, f"{test_model.name}.err").is_symlink()

    with open(pathlib.Path(test_model.path, f"{test_model.name}.out"), "r") as file:
        log_contents = file.read()

    assert "spam eggs" in log_contents

    first_link = os.readlink(pathlib.Path(test_model.path, f"{test_model.name}.out"))

    test_model.run_settings.exe_args = ["hello", "world"]
    exp.generate(test_model, overwrite=True)
    exp.start(test_model, block=True)

    assert pathlib.Path(test_model.path).exists()
    assert pathlib.Path(test_model.path, f"{test_model.name}.out").is_symlink()
    assert pathlib.Path(test_model.path, f"{test_model.name}.err").is_symlink()

    with open(pathlib.Path(test_model.path, f"{test_model.name}.out"), "r") as file:
        log_contents = file.read()

    assert "hello world" in log_contents

    second_link = os.readlink(pathlib.Path(test_model.path, f"{test_model.name}.out"))

    with open(first_link, "r") as file:
        first_historical_log = file.read()

    assert "spam eggs" in first_historical_log

    with open(second_link, "r") as file:
        second_historical_log = file.read()

    assert "hello world" in second_historical_log


def test_get_output_files_with_create_job_step(test_dir):
    """Testing output files through _create_job_step"""
    exp_dir = pathlib.Path(test_dir)
    status_dir = exp_dir / CONFIG.telemetry_subdir / model.type
    step = controller._create_job_step(model, status_dir)
    expected_out_path = status_dir / model.name / (model.name + ".out")
    expected_err_path = status_dir / model.name / (model.name + ".err")
    assert step.get_output_files() == (str(expected_out_path), str(expected_err_path))


@pytest.mark.parametrize(
    "entity",
    [pytest.param(ens, id="ensemble"), pytest.param(orc, id="orchestrator")],
)
def test_get_output_files_with_create_batch_job_step(entity, test_dir):
    """Testing output files through _create_batch_job_step"""
    exp_dir = pathlib.Path(test_dir)
    status_dir = exp_dir / CONFIG.telemetry_subdir / entity.type
    batch_step, substeps = slurm_controller._create_batch_job_step(entity, status_dir)
    for step in substeps:
        # example output path for a member of an Ensemble is
        # .smartsim/telemetry/Ensemble/ens/ens_0/ens_0.out
        expected_out_path = (
            status_dir / entity.name / step.entity_name / (step.entity_name + ".out")
        )
        expected_err_path = (
            status_dir / entity.name / step.entity_name / (step.entity_name + ".err")
        )
        assert step.get_output_files() == (
            str(expected_out_path),
            str(expected_err_path),
        )


def test_model_get_output_files(test_dir):
    """Testing model output files with manual step creation"""
    exp_dir = pathlib.Path(test_dir)
    step = Step(model.name, model.path, model.run_settings)
    step.meta["status_dir"] = exp_dir / "output_dir"
    expected_out_path = step.meta["status_dir"] / (model.name + ".out")
    expected_err_path = step.meta["status_dir"] / (model.name + ".err")
    assert step.get_output_files() == (str(expected_out_path), str(expected_err_path))


def test_ensemble_get_output_files(test_dir):
    """Testing ensemble output files with manual step creation"""
    exp_dir = pathlib.Path(test_dir)
    for member in ens.models:
        step = Step(member.name, member.path, member.run_settings)
        step.meta["status_dir"] = exp_dir / "output_dir"
        expected_out_path = step.meta["status_dir"] / (member.name + ".out")
        expected_err_path = step.meta["status_dir"] / (member.name + ".err")
        assert step.get_output_files() == (
            str(expected_out_path),
            str(expected_err_path),
        )


def test_get_output_files_no_status_dir(test_dir):
    """Test that a step not having a status directory throws a KeyError"""
    step_settings = RunSettings("echo")
    step = Step("mock-step", test_dir, step_settings)
    with pytest.raises(KeyError):
        out, err = step.get_output_files()
