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
from smartsim._core.control.manifest import LaunchedManifestBuilder
from smartsim._core.launcher.step import SbatchStep, SrunStep
from smartsim.entity import Ensemble, Model
from smartsim.error import EntityExistsError, SSUnsupportedError
from smartsim.settings import RunSettings, SbatchSettings, SrunSettings
from smartsim.settings.mpiSettings import _BaseMPISettings

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.slow_tests


def test_register_incoming_entity_preexists():
    exp = Experiment("experiment", launcher="local")
    rs = RunSettings("python", exe_args="sleep.py")
    ensemble = exp.create_ensemble(name="ensemble", replicas=1, run_settings=rs)
    m = exp.create_model("model", run_settings=rs)
    m.register_incoming_entity(ensemble["ensemble_0"])
    assert len(m.incoming_entities) == 1
    with pytest.raises(EntityExistsError):
        m.register_incoming_entity(ensemble["ensemble_0"])


def test_disable_key_prefixing():
    exp = Experiment("experiment", launcher="local")
    rs = RunSettings("python", exe_args="sleep.py")
    m = exp.create_model("model", run_settings=rs)
    m.disable_key_prefixing()
    assert m.query_key_prefixing() == False


def test_catch_colo_mpmd_model():
    exp = Experiment("experiment", launcher="local")
    rs = _BaseMPISettings("python", exe_args="sleep.py", fail_if_missing_exec=False)

    # make it an mpmd model
    rs_2 = _BaseMPISettings("python", exe_args="sleep.py", fail_if_missing_exec=False)
    rs.make_mpmd(rs_2)

    model = exp.create_model("bad_colo_model", rs)

    # make it colocated which should raise and error
    with pytest.raises(SSUnsupportedError):
        model.colocate_db()


def test_attach_batch_settings_to_model():
    exp = Experiment("experiment", launcher="slurm")
    bs = SbatchSettings()
    rs = SrunSettings("python", exe_args="sleep.py")

    model_wo_bs = exp.create_model("test_model", run_settings=rs)
    assert model_wo_bs.batch_settings is None

    model_w_bs = exp.create_model("test_model_2", run_settings=rs, batch_settings=bs)
    assert isinstance(model_w_bs.batch_settings, SbatchSettings)


@pytest.fixture
def monkeypatch_exp_controller(monkeypatch):
    def _monkeypatch_exp_controller(exp):
        entity_steps = []

        def start_wo_job_manager(
            self, exp_name, exp_path, manifest, block=True, kill_on_interrupt=True
        ):
            self._launch(exp_name, exp_path, manifest)
            return LaunchedManifestBuilder("name", "path", "launcher").finalize()

        def launch_step_nop(self, step, entity):
            entity_steps.append((step, entity))

        monkeypatch.setattr(
            exp._control,
            "start",
            start_wo_job_manager.__get__(exp._control, type(exp._control)),
        )
        monkeypatch.setattr(
            exp._control,
            "_launch_step",
            launch_step_nop.__get__(exp._control, type(exp._control)),
        )

        return entity_steps

    return _monkeypatch_exp_controller


def test_model_with_batch_settings_makes_batch_step(monkeypatch_exp_controller):
    exp = Experiment("experiment", launcher="slurm")
    bs = SbatchSettings()
    rs = SrunSettings("python", exe_args="sleep.py")
    model = exp.create_model("test_model", run_settings=rs, batch_settings=bs)

    entity_steps = monkeypatch_exp_controller(exp)
    exp.start(model)

    assert len(entity_steps) == 1
    step, entity = entity_steps[0]
    assert isinstance(entity, Model)
    assert isinstance(step, SbatchStep)


def test_model_without_batch_settings_makes_run_step(
    monkeypatch, monkeypatch_exp_controller
):
    exp = Experiment("experiment", launcher="slurm")
    rs = SrunSettings("python", exe_args="sleep.py")
    model = exp.create_model("test_model", run_settings=rs)

    # pretend we are in an allocation to not raise alloc err
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    entity_steps = monkeypatch_exp_controller(exp)
    exp.start(model)

    assert len(entity_steps) == 1
    step, entity = entity_steps[0]
    assert isinstance(entity, Model)
    assert isinstance(step, SrunStep)


def test_models_batch_settings_are_ignored_in_ensemble(monkeypatch_exp_controller):
    exp = Experiment("experiment", launcher="slurm")
    bs_1 = SbatchSettings(nodes=5)
    rs = SrunSettings("python", exe_args="sleep.py")
    model = exp.create_model("test_model", run_settings=rs, batch_settings=bs_1)

    bs_2 = SbatchSettings(nodes=10)
    ens = exp.create_ensemble("test_ensemble", batch_settings=bs_2)
    ens.add_model(model)

    entity_steps = monkeypatch_exp_controller(exp)
    exp.start(ens)

    assert len(entity_steps) == 1
    step, entity = entity_steps[0]
    assert isinstance(entity, Ensemble)
    assert isinstance(step, SbatchStep)
    assert step.batch_settings.batch_args["nodes"] == "10"
    assert len(step.step_cmds) == 1
    step_cmd = step.step_cmds[0]
    assert any("srun" in tok for tok in step_cmd)  # call the model using run settings
    assert not any("sbatch" in tok for tok in step_cmd)  # no sbatch in sbatch
