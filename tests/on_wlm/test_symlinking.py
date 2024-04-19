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


def test_batch_model_and_ensemble(test_dir):
    exp_name = "test-batch"
    exp = Experiment(exp_name, launcher="slurm", exp_path=test_dir)

    test_model = exp.create_model(
        "test_model", path=test_dir, run_settings=batch_rs, batch_settings=bs
    )
    exp.generate(test_model)
    exp.start(test_model, block=True)

    assert pathlib.Path(test_model.path).exists()
    _should_be_symlinked(pathlib.Path(test_model.path, f"{test_model.name}.out"), True)
    _should_be_symlinked(pathlib.Path(test_model.path, f"{test_model.name}.err"), False)
    _should_not_be_symlinked(pathlib.Path(test_model.path, f"{test_model.name}.sh"))

    test_ensemble = exp.create_ensemble(
        "test_ensemble", params={}, batch_settings=bs, run_settings=batch_rs, replicas=3
    )
    exp.generate(test_ensemble)
    exp.start(test_ensemble, block=True)

    assert pathlib.Path(test_ensemble.path).exists()
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


def test_batch_ensemble_symlinks(test_dir):
    exp_name = "test-batch-ensemble"
    exp = Experiment(exp_name, launcher="slurm", exp_path=test_dir)
    test_ensemble = exp.create_ensemble(
        "test_ensemble", params={}, batch_settings=bs, run_settings=batch_rs, replicas=3
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


def test_batch_model_symlinks(test_dir):
    exp_name = "test-batch-model"
    exp = Experiment(exp_name, launcher="slurm", exp_path=test_dir)

    test_model = exp.create_model(
        "test_model", path=test_dir, run_settings=batch_rs, batch_settings=bs
    )
    exp.generate(test_model)
    exp.start(test_model, block=True)

    assert pathlib.Path(test_model.path).exists()

    _should_be_symlinked(pathlib.Path(test_model.path, f"{test_model.name}.out"), True)
    _should_be_symlinked(pathlib.Path(test_model.path, f"{test_model.name}.err"), False)
    _should_not_be_symlinked(pathlib.Path(test_model.path, f"{test_model.name}.sh"))


def test_batch_orchestrator_symlinks(test_dir):
    exp = Experiment("test-batch-orc", launcher="slurm", exp_path=test_dir)
    port = 2424
    db = exp.create_database(db_nodes=3, port=port, batch=True, single_cmd=False)
    exp.generate(db)
    exp.start(db, block=True)
    exp.stop(db)

    _should_be_symlinked(pathlib.Path(db.path, f"{db.name}.out"), False)
    _should_be_symlinked(pathlib.Path(db.path, f"{db.name}.err"), False)

    for i in range(db.db_nodes):
        _should_be_symlinked(pathlib.Path(db.path, f"{db.name}_{i}.out"), False)
        _should_be_symlinked(pathlib.Path(db.path, f"{db.name}_{i}.err"), False)
        _should_not_be_symlinked(
            pathlib.Path(db.path, f"nodes-orchestrator_{i}-{port}.conf")
        )


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
