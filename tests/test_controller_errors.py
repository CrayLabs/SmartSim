# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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
from smartsim.settings.slurmSettings import SrunSettings
from smartsim._core.control import Controller, Manifest
from smartsim.database import Orchestrator
from smartsim.entity import Model
from smartsim.error import SmartSimError, SSUnsupportedError
from smartsim.error.errors import SSUnsupportedError
from smartsim.settings import RunSettings
from smartsim.entity.ensemble import Ensemble

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_finished_entity_orc_error():
    """Orchestrators are never 'finished', either run forever or stopped by user"""
    orc = Orchestrator()
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.finished(orc)


def test_finished_entity_wrong_type():
    """Wrong type supplied to controller.finished"""
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.finished([])


def test_finished_not_found():
    """Ask if model is finished that hasnt been launched by this experiment"""
    rs = RunSettings("python")
    model = Model("hello", {}, "./", rs)
    cont = Controller(launcher="local")
    with pytest.raises(ValueError):
        cont.finished(model)


def test_entity_status_wrong_type():
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.get_entity_status([])


def test_entity_list_status_wrong_type():
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.get_entity_list_status([])


def test_unsupported_launcher():
    """Test when user provideds unsupported launcher"""
    cont = Controller(launcher="local")
    with pytest.raises(SSUnsupportedError):
        cont.init_launcher("thelauncherwhichdoesnotexist")


def test_no_launcher():
    """Test when user provideds unsupported launcher"""
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.init_launcher(None)


def test_wrong_orchestrator(wlmutils):
    # lo interface to avoid warning from SmartSim
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        interface="lo",
        run_command="aprun",
        launcher="pbs",
    )
    cont = Controller(launcher="local")
    manifest = Manifest(orc)
    with pytest.raises(SmartSimError):
        cont._launch("exp_name", "exp_path", manifest)


def test_bad_orc_checkpoint():
    checkpoint = "./bad-checkpoint"
    cont = Controller(launcher="local")
    with pytest.raises(FileNotFoundError):
        cont.reload_saved_db(checkpoint)

controller = Controller()
rs = SrunSettings("echo", ["spam", "eggs"])
model = Model("duplicate", run_settings=rs, params={}, path="")
ens = Ensemble("duplicate", params={}, run_settings=rs, replicas=2)
orc = Orchestrator("duplicate", db_nodes=3, batch=True, launcher="slurm", run_command="srun")

@pytest.mark.parametrize(
    "collection",
    [
        pytest.param(ens, id="Ensemble"),
        pytest.param(orc, id="Database"),
        pytest.param(model, id="Model"),
    ],
)
def test_duplicate_entity(wlmutils, collection):
    test_launcher = wlmutils.get_test_launcher()
    controller = Controller(test_launcher)
    controller._jobs.add_job("duplicate", job_id="1234", entity=collection)
    print(f"after: {controller._jobs.jobs}")
    controller._jobs.move_to_completed(controller._jobs.jobs.get("duplicate"))
    print(f"this is printed: {controller._jobs.completed}")
    #controller._launch_step("model_1", entity=model)

# so I need to test local and WLM launchers
# I need to test model, ensemble and orch
# I need to test running model, ensemble and orch
# I need to test completed model, emsemble and orch