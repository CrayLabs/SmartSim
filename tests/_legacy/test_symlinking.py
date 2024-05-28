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


@pytest.mark.parametrize(
    "entity",
    [pytest.param(ens, id="ensemble"), pytest.param(model, id="model")],
)
def test_symlink(test_dir, entity):
    """Test symlinking historical output files"""
    entity.path = test_dir
    if entity.type == Ensemble:
        for member in ens.models:
            symlink_with_create_job_step(test_dir, member)
    else:
        symlink_with_create_job_step(test_dir, entity)


def symlink_with_create_job_step(test_dir, entity):
    """Function that helps cut down on repeated testing code"""
    exp_dir = pathlib.Path(test_dir)
    entity.path = test_dir
    status_dir = exp_dir / CONFIG.telemetry_subdir / entity.type
    step = controller._create_job_step(entity, status_dir)
    controller.symlink_output_files(step, entity)
    assert pathlib.Path(entity.path, f"{entity.name}.out").is_symlink()
    assert pathlib.Path(entity.path, f"{entity.name}.err").is_symlink()
    assert os.readlink(pathlib.Path(entity.path, f"{entity.name}.out")) == str(
        status_dir / entity.name / (entity.name + ".out")
    )
    assert os.readlink(pathlib.Path(entity.path, f"{entity.name}.err")) == str(
        status_dir / entity.name / (entity.name + ".err")
    )


@pytest.mark.parametrize(
    "entity",
    [
        pytest.param(ens, id="ensemble"),
        pytest.param(orc, id="orchestrator"),
        pytest.param(anon_batch_model, id="model"),
    ],
)
def test_batch_symlink(entity, test_dir):
    """Test symlinking historical output files"""
    exp_dir = pathlib.Path(test_dir)
    entity.path = test_dir
    status_dir = exp_dir / CONFIG.telemetry_subdir / entity.type
    batch_step, substeps = slurm_controller._create_batch_job_step(entity, status_dir)
    for step in substeps:
        slurm_controller.symlink_output_files(step, entity)
        assert pathlib.Path(entity.path, f"{entity.name}.out").is_symlink()
        assert pathlib.Path(entity.path, f"{entity.name}.err").is_symlink()
        assert os.readlink(pathlib.Path(entity.path, f"{entity.name}.out")) == str(
            status_dir / entity.name / step.entity_name / (step.entity_name + ".out")
        )
        assert os.readlink(pathlib.Path(entity.path, f"{entity.name}.err")) == str(
            status_dir / entity.name / step.entity_name / (step.entity_name + ".err")
        )


def test_symlink_error(test_dir):
    """Ensure FileNotFoundError is thrown"""
    bad_model = Model(
        "bad_model",
        params={},
        path=pathlib.Path(test_dir, "badpath"),
        run_settings=RunSettings("echo"),
    )
    telem_dir = pathlib.Path(test_dir, "bad_model_telemetry")
    bad_step = controller._create_job_step(bad_model, telem_dir)
    with pytest.raises(FileNotFoundError):
        controller.symlink_output_files(bad_step, bad_model)


def test_failed_model_launch_symlinks(test_dir):
    exp_name = "failed-exp"
    exp = Experiment(exp_name, exp_path=test_dir)
    test_model = exp.create_model(
        "test_model", run_settings=batch_rs, batch_settings=bs
    )
    exp.generate(test_model)
    with pytest.raises(TypeError):
        exp.start(test_model)

    _should_not_be_symlinked(pathlib.Path(test_model.path))
    assert not pathlib.Path(test_model.path, f"{test_model.name}.out").is_symlink()
    assert not pathlib.Path(test_model.path, f"{test_model.name}.err").is_symlink()


def test_failed_ensemble_launch_symlinks(test_dir):
    exp_name = "failed-exp"
    exp = Experiment(exp_name, exp_path=test_dir)
    test_ensemble = exp.create_ensemble(
        "test_ensemble", params={}, batch_settings=bs, run_settings=batch_rs, replicas=3
    )
    exp.generate(test_ensemble)
    with pytest.raises(TypeError):
        exp.start(test_ensemble)

    _should_not_be_symlinked(pathlib.Path(test_ensemble.path))
    assert not pathlib.Path(
        test_ensemble.path, f"{test_ensemble.name}.out"
    ).is_symlink()
    assert not pathlib.Path(
        test_ensemble.path, f"{test_ensemble.name}.err"
    ).is_symlink()

    for i in range(len(test_ensemble.models)):
        assert not pathlib.Path(
            test_ensemble.path,
            f"{test_ensemble.name}_{i}",
            f"{test_ensemble.name}_{i}.out",
        ).is_symlink()
        assert not pathlib.Path(
            test_ensemble.path,
            f"{test_ensemble.name}_{i}",
            f"{test_ensemble.name}_{i}.err",
        ).is_symlink()


def test_non_batch_ensemble_symlinks(test_dir):
    exp_name = "test-non-batch-ensemble"
    rs = RunSettings("echo", ["spam", "eggs"])
    exp = Experiment(exp_name, exp_path=test_dir)
    test_ensemble = exp.create_ensemble(
        "test_ensemble", params={}, run_settings=rs, replicas=3
    )
    exp.generate(test_ensemble)
    exp.start(test_ensemble, block=True)

    for i in range(len(test_ensemble.models)):
        _should_be_symlinked(
            pathlib.Path(
                test_ensemble.path,
                f"{test_ensemble.name}_{i}",
                f"{test_ensemble.name}_{i}.out",
            ),
            True,
        )
        _should_be_symlinked(
            pathlib.Path(
                test_ensemble.path,
                f"{test_ensemble.name}_{i}",
                f"{test_ensemble.name}_{i}.err",
            ),
            False,
        )

    _should_not_be_symlinked(pathlib.Path(exp.exp_path, "smartsim_params.txt"))


def test_non_batch_model_symlinks(test_dir):
    exp_name = "test-non-batch-model"
    exp = Experiment(exp_name, exp_path=test_dir)
    rs = RunSettings("echo", ["spam", "eggs"])

    test_model = exp.create_model("test_model", path=test_dir, run_settings=rs)
    exp.generate(test_model)
    exp.start(test_model, block=True)

    assert pathlib.Path(test_model.path).exists()

    _should_be_symlinked(pathlib.Path(test_model.path, f"{test_model.name}.out"), True)
    _should_be_symlinked(pathlib.Path(test_model.path, f"{test_model.name}.err"), False)
    _should_not_be_symlinked(pathlib.Path(exp.exp_path, "smartsim_params.txt"))


def test_non_batch_orchestrator_symlinks(test_dir):
    exp = Experiment("test-non-batch-orc", exp_path=test_dir)

    db = exp.create_database(interface="lo")
    exp.generate(db)
    exp.start(db, block=True)
    exp.stop(db)

    for i in range(db.db_nodes):
        _should_be_symlinked(pathlib.Path(db.path, f"{db.name}_{i}.out"), False)
        _should_be_symlinked(pathlib.Path(db.path, f"{db.name}_{i}.err"), False)

    _should_not_be_symlinked(pathlib.Path(exp.exp_path, "smartsim_params.txt"))


def _should_not_be_symlinked(non_linked_path: pathlib.Path):
    """Helper function for assertions about paths that should NOT be symlinked"""
    assert non_linked_path.exists()
    assert not non_linked_path.is_symlink()


def _should_be_symlinked(linked_path: pathlib.Path, open_file: bool):
    """Helper function for assertions about paths that SHOULD be symlinked"""
    assert linked_path.exists()
    assert linked_path.is_symlink()
    # ensure the source file exists
    assert pathlib.Path(os.readlink(linked_path)).exists()
    if open_file:
        with open(pathlib.Path(os.readlink(linked_path)), "r") as file:
            log_contents = file.read()
        assert "spam eggs" in log_contents
