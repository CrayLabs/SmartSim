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

from smartsim._core.control import Controller, Manifest
from smartsim._core.launcher.step import Step
from smartsim._core.launcher.step.dragon_step import DragonStep
from smartsim.builders.ensemble import Ensemble
from smartsim.database import FeatureStore
from smartsim.entity import Application
from smartsim.error import SmartSimError, SSUnsupportedError
from smartsim.error.errors import SSUnsupportedError
from smartsim.settings import RunSettings, SrunSettings

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a

entity_settings = SrunSettings("echo", ["spam", "eggs"])
application_dup_setting = RunSettings("echo", ["spam_1", "eggs_2"])
application = Application(
    "application_name", run_settings=entity_settings, params={}, path=""
)
# Application entity slightly different but with same name
application_2 = Application(
    "application_name", run_settings=application_dup_setting, params={}, path=""
)
ens = Ensemble("ensemble_name", params={}, run_settings=entity_settings, replicas=2)
# Ensemble entity slightly different but with same name
ens_2 = Ensemble("ensemble_name", params={}, run_settings=entity_settings, replicas=3)
feature_store = FeatureStore(
    fs_nodes=3, batch=True, launcher="slurm", run_command="srun"
)


def test_finished_entity_feature_store_error():
    """FeatureStores are never 'finished', either run forever or stopped by user"""
    feature_store = FeatureStore()
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.finished(feature_store)


def test_finished_entity_wrong_type():
    """Wrong type supplied to controller.finished"""
    cont = Controller(launcher="local")
    with pytest.raises(TypeError):
        cont.finished([])


def test_finished_not_found():
    """Ask if application is finished that hasnt been launched by this experiment"""
    rs = RunSettings("python")
    application = Application("hello", {}, "./", rs)
    cont = Controller(launcher="local")
    with pytest.raises(ValueError):
        cont.finished(application)


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


def test_wrong_feature_store(wlmutils):
    # lo interface to avoid warning from SmartSim
    feature_store = FeatureStore(
        wlmutils.get_test_port(),
        fs_nodes=3,
        interface="lo",
        run_command="aprun",
        launcher="pbs",
    )
    cont = Controller(launcher="local")
    manifest = Manifest(feature_store)
    with pytest.raises(SmartSimError):
        cont._launch("exp_name", "exp_path", manifest)


def test_bad_feature_store_checkpoint():
    checkpoint = "./bad-checkpoint"
    cont = Controller(launcher="local")
    with pytest.raises(FileNotFoundError):
        cont.reload_saved_fs(checkpoint)


class MockStep(Step):
    """Mock step to implement any abstract methods so that it can be
    instanced for test purposes
    """

    def get_launch_cmd(self):
        return ["echo", "spam"]


@pytest.mark.parametrize(
    "entity",
    [
        pytest.param(ens, id="Ensemble_running"),
        pytest.param(application, id="Application_running"),
        pytest.param(orc, id="Feature_store_running"),
    ],
)
def test_duplicate_running_entity(test_dir, wlmutils, entity):
    """This test validates that users cannot reuse entity names
    that are running in JobManager.jobs or JobManager.fs_jobs
    """
    step_settings = RunSettings("echo")
    step = MockStep("mock-step", test_dir, step_settings)
    test_launcher = wlmutils.get_test_launcher()
    controller = Controller(test_launcher)
    controller._jobs.add_job(entity.name, job_id="1234", entity=entity)
    with pytest.raises(SSUnsupportedError) as ex:
        controller._launch_step(step, entity=entity)
    assert ex.value.args[0] == "SmartSim entities cannot have duplicate names."


@pytest.mark.parametrize(
    "entity",
    [
        pytest.param(ens, id="Ensemble_running"),
        pytest.param(application, id="Application_running"),
    ],
)
def test_restarting_entity(test_dir, wlmutils, entity):
    """Validate restarting a completed Application/Ensemble job"""
    step_settings = RunSettings("echo")
    test_launcher = wlmutils.get_test_launcher()
    step = MockStep("mock-step", test_dir, step_settings)
    step.meta["status_dir"] = test_dir
    entity.path = test_dir
    controller = Controller(test_launcher)
    controller._jobs.add_job(entity.name, job_id="1234", entity=entity)
    controller._jobs.move_to_completed(controller._jobs.jobs.get(entity.name))
    controller._launch_step(step, entity=entity)


def test_restarting_feature_store(test_dir, wlmutils):
    """Validate restarting a completed FeatureStore job"""
    step_settings = RunSettings("echo")
    test_launcher = wlmutils.get_test_launcher()
    step = MockStep("mock-step", test_dir, step_settings)
    step.meta["status_dir"] = test_dir
    feature_store.path = test_dir
    controller = Controller(test_launcher)
    controller._jobs.add_job(feature_store.name, job_id="1234", entity=feature_store)
    controller._jobs.move_to_completed(controller._jobs.fs_jobs.get(feature_store.name))
    controller._launch_step(step, entity=feature_store)


@pytest.mark.parametrize(
    "entity,entity_2",
    [
        pytest.param(ens, ens_2, id="Ensemble_running"),
        pytest.param(application, application_2, id="Application_running"),
    ],
)
def test_starting_entity(test_dir, wlmutils, entity, entity_2):
    """Test launching a job of Application/Ensemble with same name in completed"""
    step_settings = RunSettings("echo")
    step = MockStep("mock-step", test_dir, step_settings)
    test_launcher = wlmutils.get_test_launcher()
    controller = Controller(test_launcher)
    controller._jobs.add_job(entity.name, job_id="1234", entity=entity)
    controller._jobs.move_to_completed(controller._jobs.jobs.get(entity.name))
    with pytest.raises(SSUnsupportedError) as ex:
        controller._launch_step(step, entity=entity_2)
    assert ex.value.args[0] == "SmartSim entities cannot have duplicate names."
