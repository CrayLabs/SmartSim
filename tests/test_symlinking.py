# BSD 2-Clause License
#
# Copyright (c) 2021-2025, Hewlett Packard Enterprise
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


@pytest.fixture
def model_entity():
    return Model("test_model", params={}, path="", run_settings=rs)


@pytest.fixture
def ensemble_entity():
    return Ensemble("ens", params={}, run_settings=rs, batch_settings=bs, replicas=3)


@pytest.fixture
def orchestrator_entity():
    return Orchestrator(
        db_nodes=3, batch=True, launcher="slurm", run_command="srun"
    )


@pytest.fixture
def anon_batch_model_entity():
    batch_model = Model(
        "batch_test_model",
        params={},
        path="",
        run_settings=batch_rs,
        batch_settings=bs,
    )
    return _AnonymousBatchJob(batch_model)


@pytest.mark.parametrize(
    "entity_fixture",
    ["ensemble_entity", "model_entity"],
    ids=["ensemble", "model"],
)
def test_symlink(test_dir, request, entity_fixture):
    """Test symlinking historical output files"""
    entity = request.getfixturevalue(entity_fixture)
    entity.path = test_dir
    if entity.type == Ensemble:
        for member in entity.models:
            symlink_with_create_job_step(test_dir, member)
    else:
        symlink_with_create_job_step(test_dir, entity)


def symlink_with_create_job_step(test_dir, entity):
    """Function that helps cut down on repeated testing code"""
    exp_dir = pathlib.Path(test_dir)
    entity.path = test_dir
    # Use consistent metadata directory structure
    metadata_dir = exp_dir / CONFIG.metadata_subdir
    step = controller._create_job_step(entity, metadata_dir)
    controller.symlink_output_files(step, entity)
    assert pathlib.Path(entity.path, f"{entity.name}.out").is_symlink()
    assert pathlib.Path(entity.path, f"{entity.name}.err").is_symlink()
    # Verify symlinks point to the correct metadata directory
    expected_out = metadata_dir / (entity.name + ".out")
    expected_err = metadata_dir / (entity.name + ".err")
    assert os.readlink(pathlib.Path(entity.path, f"{entity.name}.out")) == str(
        expected_out
    )
    assert os.readlink(pathlib.Path(entity.path, f"{entity.name}.err")) == str(
        expected_err
    )


@pytest.mark.parametrize(
    "entity_fixture",
    [
        "ensemble_entity",
        "orchestrator_entity",
        "anon_batch_model_entity",
    ],
    ids=["ensemble", "orchestrator", "model"],
)
def test_batch_symlink(request, entity_fixture, test_dir):
    """Test symlinking historical output files"""
    entity = request.getfixturevalue(entity_fixture)
    exp_dir = pathlib.Path(test_dir)
    entity.path = test_dir
    # For entities with sub-entities (like Orchestrator), set their paths too
    if hasattr(entity, "entities"):
        for sub_entity in entity.entities:
            sub_entity.path = test_dir

    # Create metadata_dir to simulate consistent metadata structure
    metadata_dir = exp_dir / CONFIG.metadata_subdir
    _, substeps = slurm_controller._create_batch_job_step(entity, metadata_dir)

    # For batch entities, we need to call symlink_output_files correctly
    # Based on how the controller does it, we should pass the individual entities
    for substep in substeps:
        # Just test the first substep and entity pair
        substep_entity = entity.entities[0]
        slurm_controller.symlink_output_files(substep, substep_entity)

        # The symlinks should be created in the substep entity's path using its name
        symlink_out = pathlib.Path(substep_entity.path, f"{substep_entity.name}.out")
        symlink_err = pathlib.Path(substep_entity.path, f"{substep_entity.name}.err")

        assert symlink_out.is_symlink()
        assert symlink_err.is_symlink()

        # The symlinks should point to the metadata_dir set for this substep
        expected_out = pathlib.Path(substep.meta["metadata_dir"]) / (
            substep.entity_name + ".out"
        )
        expected_err = pathlib.Path(substep.meta["metadata_dir"]) / (
            substep.entity_name + ".err"
        )

        assert os.readlink(symlink_out) == str(expected_out)
        assert os.readlink(symlink_err) == str(expected_err)

        # For _AnonymousBatchJob (single model)
        slurm_controller.symlink_output_files(substep, entity)

        symlink_out = pathlib.Path(entity.path, f"{entity.name}.out")
        symlink_err = pathlib.Path(entity.path, f"{entity.name}.err")

        assert symlink_out.is_symlink()
        assert symlink_err.is_symlink()


def test_symlink_error(test_dir):
    """Ensure FileNotFoundError is thrown"""
    bad_model = Model(
        "bad_model",
        params={},
        path=pathlib.Path(test_dir, "badpath"),
        run_settings=RunSettings("echo"),
    )
    metadata_dir = pathlib.Path(test_dir, "bad_model_metadata")
    bad_step = controller._create_job_step(bad_model, metadata_dir)
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
