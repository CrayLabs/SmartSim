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

import pathlib

import pytest
import os


from smartsim import Experiment
from smartsim._core.config import CONFIG
from smartsim._core.control.controller import Controller
from smartsim._core.launcher.step import Step
from smartsim.database.orchestrator import Orchestrator
from smartsim.entity.ensemble import Ensemble
from smartsim.entity.model import Model
from smartsim.settings.base import RunSettings
from smartsim.settings.slurmSettings import SbatchSettings, SrunSettings

controller = Controller()
slurm_controller = Controller(launcher="slurm")

rs = SrunSettings("echo", ["spam", "eggs"])
bs = SbatchSettings()

ens = Ensemble("ens", params={}, run_settings=rs, batch_settings=bs, replicas=3)
orc = Orchestrator(db_nodes=3, batch=True, launcher="slurm", run_command="srun")


@pytest.mark.parametrize(
    "entity",
    [
        pytest.param(Model("test_model", {}, "", rs)),
        pytest.param(ens),

    ],
)
def test_get_output_files_with_create_job_step(entity, test_dir):
    exp_dir = pathlib.Path(test_dir)
    status_dir = exp_dir / CONFIG.telemetry_subdir / entity.type
    step = controller._create_job_step(entity, status_dir)
    expected_out_path = status_dir / entity.name / (entity.name + ".out")
    expected_err_path = status_dir / entity.name / (entity.name + ".err")
    assert step.get_output_files() == (str(expected_out_path), str(expected_err_path))


def test_get_output_files_no_status_dir(test_dir):
    step_settings = RunSettings("echo")
    step = Step("mock-step", test_dir, step_settings)
    with pytest.raises(KeyError):
        out, err = step.get_output_files()


@pytest.mark.parametrize(
    "entity",
    [
        pytest.param(Model("test_model", {}, "", rs)),
        pytest.param(ens),

    ],
)
def test_symlink(entity, test_dir):
    exp_dir = pathlib.Path(test_dir)
    status_dir = exp_dir / CONFIG.telemetry_subdir / entity.type
    step = controller._create_job_step(entity, status_dir)
    controller.symlink(step, entity)
    assert os.path.islink(os.path.join(entity.path, f"{entity.name}.out"))
    assert os.path.islink(os.path.join(entity.path, f"{entity.name}.err"))
    assert os.readlink(os.path.join(entity.path, f"{entity.name}.out")) ==  str(status_dir / entity.name / (entity.name + ".out"))
    assert os.readlink(os.path.join(entity.path, f"{entity.name}.err")) ==  str(status_dir / entity.name / (entity.name + ".err"))
