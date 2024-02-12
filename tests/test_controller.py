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

from smartsim._core.control.controller import Controller
from smartsim._core.launcher.step import Step
from smartsim.database.orchestrator import Orchestrator
from smartsim.entity.ensemble import Ensemble
from smartsim.settings.slurmSettings import SbatchSettings, SrunSettings

controller = Controller()

rs = SrunSettings("echo", ["spam", "eggs"])
bs = SbatchSettings()

ens = Ensemble("ens", params={}, run_settings=rs, batch_settings=bs, replicas=3)
orc = Orchestrator(db_nodes=3, batch=True, launcher="slurm", run_command="srun")


class MockStep(Step):
    @staticmethod
    def _create_unique_name(name):
        return name

    def add_to_batch(self, step): ...

    def get_launch_cmd(self):
        return []


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param(ens, id="Ensemble"),
        pytest.param(orc, id="Database"),
    ],
)
def test_controller_batch_step_creation_preserves_entity_order(collection, monkeypatch):
    monkeypatch.setattr(
        controller._launcher,
        "create_step",
        lambda name, path, settings: MockStep(name, path, settings),
    )
    entity_names = [x.name for x in collection.entities]
    assert len(entity_names) == len(set(entity_names))
    _, steps = controller._create_batch_job_step(
        collection, pathlib.Path("mock/exp/path")
    )
    assert entity_names == [step.name for step in steps]
