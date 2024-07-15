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

import json
import pathlib
import shutil
import sys
import typing as t

import pytest

from smartsim._core.launcher.step.dragonStep import DragonBatchStep, DragonStep
from smartsim.settings import DragonRunSettings
from smartsim.settings.pbsSettings import QsubBatchSettings
from smartsim.settings.slurmSettings import SbatchSettings

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


from smartsim._core.schemas.dragonRequests import *
from smartsim._core.schemas.dragonResponses import *


@pytest.fixture
def dragon_batch_step(test_dir: str) -> DragonBatchStep:
    """Fixture for creating a default batch of steps for a dragon launcher"""
    test_path = pathlib.Path(test_dir)

    batch_step_name = "batch_step"
    num_nodes = 4
    batch_settings = SbatchSettings(nodes=num_nodes)
    batch_step = DragonBatchStep(batch_step_name, test_dir, batch_settings)

    # ensure the status_dir is set
    status_dir = (test_path / ".smartsim" / "logs").as_posix()
    batch_step.meta["status_dir"] = status_dir

    # create some steps to verify the requests file output changes
    rs0 = DragonRunSettings(exe="sleep", exe_args=["1"])
    rs1 = DragonRunSettings(exe="sleep", exe_args=["2"])
    rs2 = DragonRunSettings(exe="sleep", exe_args=["3"])
    rs3 = DragonRunSettings(exe="sleep", exe_args=["4"])

    names = "test00", "test01", "test02", "test03"
    settings = rs0, rs1, rs2, rs3

    # create steps with:
    # no affinity, cpu affinity only, gpu affinity only, cpu and gpu affinity
    cpu_affinities = [[], [0, 1, 2], [], [3, 4, 5, 6]]
    gpu_affinities = [[], [], [0, 1, 2], [3, 4, 5, 6]]

    # assign some unique affinities to each run setting instance
    for index, rs in enumerate(settings):
        if gpu_affinities[index]:
            rs.set_node_feature("gpu")
        rs.set_cpu_affinity(cpu_affinities[index])
        rs.set_gpu_affinity(gpu_affinities[index])

    steps = list(
        DragonStep(name_, test_dir, rs_) for name_, rs_ in zip(names, settings)
    )

    for index, step in enumerate(steps):
        # ensure meta is configured...
        step.meta["status_dir"] = status_dir
        # ... and put all the steps into the batch
        batch_step.add_to_batch(steps[index])

    return batch_step


def get_request_path_from_batch_script(launch_cmd: t.List[str]) -> pathlib.Path:
    """Helper method for finding the path to a request file from the launch command"""
    script_path = pathlib.Path(launch_cmd[-1])
    batch_script = script_path.read_text(encoding="utf-8")
    batch_statements = [line for line in batch_script.split("\n") if line]
    entrypoint_cmd = batch_statements[-1]
    requests_file = pathlib.Path(entrypoint_cmd.split()[-1])
    return requests_file


def test_dragon_step_creation(test_dir: str) -> None:
    """Verify that the step is created with the values provided"""
    rs = DragonRunSettings(exe="sleep", exe_args=["1"])

    original_name = "test"
    step = DragonStep(original_name, test_dir, rs)

    # confirm the name has been made unique to avoid conflicts
    assert step.name != original_name
    assert step.entity_name == original_name
    assert step.cwd == test_dir
    assert step.step_settings is not None


def test_dragon_step_name_uniqueness(test_dir: str) -> None:
    """Verify that step name is unique and independent of step content"""

    rs = DragonRunSettings(exe="sleep", exe_args=["1"])

    original_name = "test"

    num_steps = 100
    steps = [DragonStep(original_name, test_dir, rs) for _ in range(num_steps)]

    # confirm the name has been made unique in each step
    step_names = {step.name for step in steps}
    assert len(step_names) == num_steps


def test_dragon_step_launch_cmd(test_dir: str) -> None:
    """Verify the expected launch cmd is generated w/minimal settings"""
    exp_exe = "sleep"
    exp_exe_args = "1"
    rs = DragonRunSettings(exe=exp_exe, exe_args=[exp_exe_args])

    original_name = "test"
    step = DragonStep(original_name, test_dir, rs)

    launch_cmd = step.get_launch_cmd()
    assert len(launch_cmd) == 2

    # we'll verify the exe_args and exe name are handled correctly
    exe, args = launch_cmd
    assert exp_exe in exe
    assert exp_exe_args in args

    # also, verify that a string exe_args param instead of list is handled correctly
    exp_exe_args = "1 2 3"
    rs = DragonRunSettings(exe=exp_exe, exe_args=exp_exe_args)
    step = DragonStep(original_name, test_dir, rs)
    launch_cmd = step.get_launch_cmd()
    assert len(launch_cmd) == 4  # "/foo/bar/sleep 1 2 3"


def test_dragon_step_launch_cmd_multi_arg(test_dir: str) -> None:
    """Verify the expected launch cmd is generated when multiple arguments
    are passed to run settings"""
    exp_exe = "sleep"
    arg0, arg1, arg2 = "1", "2", "3"
    rs = DragonRunSettings(exe=exp_exe, exe_args=[arg0, arg1, arg2])

    original_name = "test"

    step = DragonStep(original_name, test_dir, rs)

    launch_cmd = step.get_launch_cmd()
    assert len(launch_cmd) == 4

    exe, *args = launch_cmd
    assert exp_exe in exe
    assert arg0 in args
    assert arg1 in args
    assert arg2 in args


def test_dragon_step_launch_cmd_no_bash(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that requirement for bash shell is checked"""
    exp_exe = "sleep"
    arg0, arg1, arg2 = "1", "2", "3"
    rs = DragonRunSettings(exe=exp_exe, exe_args=[arg0, arg1, arg2])
    rs.colocated_db_settings = {"foo": "bar"}  # triggers bash lookup

    original_name = "test"
    step = DragonStep(original_name, test_dir, rs)

    with pytest.raises(RuntimeError) as ex, monkeypatch.context() as ctx:
        ctx.setattr(shutil, "which", lambda _: None)
        step.get_launch_cmd()

    # verify the exception thrown is the one we're looking for
    assert "Could not find" in ex.value.args[0]


def test_dragon_step_colocated_db() -> None:
    # todo: implement a test for the branch where bash is found and
    # run_settings.colocated_db_settings is set
    ...


def test_dragon_step_container() -> None:
    # todo: implement a test for the branch where run_settings.container
    # is an instance of class `Singularity`
    ...


def test_dragon_step_run_settings_accessor(test_dir: str) -> None:
    """Verify the run settings passed to the step are copied correctly and
    are not inadvertently modified outside the step"""
    exp_exe = "sleep"
    arg0, arg1, arg2 = "1", "2", "3"
    rs = DragonRunSettings(exe=exp_exe, exe_args=[arg0, arg1, arg2])

    original_name = "test"
    step = DragonStep(original_name, test_dir, rs)
    rs_output = step.run_settings

    assert rs.exe == rs_output.exe
    assert rs.exe_args == rs_output.exe_args

    # ensure we have a deep copy
    rs.exe = "foo"
    assert id(step.run_settings) != id(rs)
    assert step.run_settings.exe != rs.exe


def test_dragon_batch_step_creation(test_dir: str) -> None:
    """Verify that the batch step is created with the values provided"""
    batch_step_name = "batch_step"
    num_nodes = 4
    batch_settings = SbatchSettings(nodes=num_nodes)
    batch_step = DragonBatchStep(batch_step_name, test_dir, batch_settings)

    # confirm the name has been made unique to avoid conflicts
    assert batch_step.name != batch_step_name
    assert batch_step.entity_name == batch_step_name
    assert batch_step.cwd == test_dir
    assert batch_step.batch_settings is not None
    assert batch_step.managed


def test_dragon_batch_step_add_to_batch(test_dir: str) -> None:
    """Verify that steps are added to the batch correctly"""
    rs = DragonRunSettings(exe="sleep", exe_args=["1"])

    name0, name1, name2 = "test00", "test01", "test02"
    step0 = DragonStep(name0, test_dir, rs)
    step1 = DragonStep(name1, test_dir, rs)
    step2 = DragonStep(name2, test_dir, rs)

    batch_step_name = "batch_step"
    num_nodes = 4
    batch_settings = SbatchSettings(nodes=num_nodes)
    batch_step = DragonBatchStep(batch_step_name, test_dir, batch_settings)

    assert len(batch_step.steps) == 0

    batch_step.add_to_batch(step0)
    assert len(batch_step.steps) == 1
    assert name0 in ",".join({step.name for step in batch_step.steps})

    batch_step.add_to_batch(step1)
    assert len(batch_step.steps) == 2
    assert name1 in ",".join({step.name for step in batch_step.steps})

    batch_step.add_to_batch(step2)
    assert len(batch_step.steps) == 3
    assert name2 in ",".join({step.name for step in batch_step.steps})


def test_dragon_batch_step_get_launch_command_meta_fail(test_dir: str) -> None:
    """Verify that the batch launch command cannot be generated without
    having the status directory set in the step metadata"""
    batch_step_name = "batch_step"
    num_nodes = 4
    batch_settings = SbatchSettings(nodes=num_nodes)
    batch_step = DragonBatchStep(batch_step_name, test_dir, batch_settings)

    with pytest.raises(KeyError) as ex:
        batch_step.get_launch_cmd()


@pytest.mark.parametrize(
    "batch_settings_class,batch_exe,batch_header,node_spec_tpl",
    [
        pytest.param(
            SbatchSettings, "sbatch", "#SBATCH", "#SBATCH --nodes={0}", id="sbatch"
        ),
        pytest.param(QsubBatchSettings, "qsub", "#PBS", "#PBS -l nodes={0}", id="qsub"),
    ],
)
def test_dragon_batch_step_get_launch_command(
    test_dir: str,
    batch_settings_class: t.Type,
    batch_exe: str,
    batch_header: str,
    node_spec_tpl: str,
) -> None:
    """Verify that the batch launch command is properly generated and
    the expected side effects are present (writing script file to disk)"""
    test_path = pathlib.Path(test_dir)

    batch_step_name = "batch_step"
    num_nodes = 4
    batch_settings = batch_settings_class(nodes=num_nodes)
    batch_step = DragonBatchStep(batch_step_name, test_dir, batch_settings)

    # ensure the status_dir is set
    status_dir = (test_path / ".smartsim" / "logs").as_posix()
    batch_step.meta["status_dir"] = status_dir

    launch_cmd = batch_step.get_launch_cmd()
    assert launch_cmd

    full_cmd = " ".join(launch_cmd)
    assert batch_exe in full_cmd  # verify launcher running the batch
    assert test_dir in full_cmd  # verify outputs are sent to expected directory
    assert "batch_step.sh" in full_cmd  # verify batch script name is in the command

    # ...verify that the script file is written when getting the launch command
    script_path = pathlib.Path(launch_cmd[-1])
    assert script_path.exists()
    assert len(script_path.read_bytes()) > 0

    batch_script = script_path.read_text(encoding="utf-8")

    # ...verify the script file has the expected batch script header content
    assert batch_header in batch_script
    assert node_spec_tpl.format(num_nodes) in batch_script  # verify node count is set

    # ...verify the script has the expected entrypoint command
    batch_statements = [line for line in batch_script.split("\n") if line]
    python_path = sys.executable

    entrypoint_cmd = batch_statements[-1]
    assert python_path in entrypoint_cmd
    assert "smartsim._core.entrypoints.dragon_client +submit" in entrypoint_cmd


def test_dragon_batch_step_write_request_file_no_steps(test_dir: str) -> None:
    """Verify that the batch launch command writes an appropriate request file
    if no steps are attached"""
    test_path = pathlib.Path(test_dir)

    batch_step_name = "batch_step"
    num_nodes = 4
    batch_settings = SbatchSettings(nodes=num_nodes)
    batch_step = DragonBatchStep(batch_step_name, test_dir, batch_settings)

    # ensure the status_dir is set
    status_dir = (test_path / ".smartsim" / "logs").as_posix()
    batch_step.meta["status_dir"] = status_dir

    launch_cmd = batch_step.get_launch_cmd()
    requests_file = get_request_path_from_batch_script(launch_cmd)

    # no steps have been added yet, so the requests file should be a serialized, empty list
    assert requests_file.read_text(encoding="utf-8") == "[]"


def test_dragon_batch_step_write_request_file(
    dragon_batch_step: DragonBatchStep,
) -> None:
    """Verify that the batch launch command writes an appropriate request file
    for the set of attached steps"""
    # create steps with:
    # no affinity, cpu affinity only, gpu affinity only, cpu and gpu affinity
    cpu_affinities = [[], [0, 1, 2], [], [3, 4, 5, 6]]
    gpu_affinities = [[], [], [0, 1, 2], [3, 4, 5, 6]]

    launch_cmd = dragon_batch_step.get_launch_cmd()
    requests_file = get_request_path_from_batch_script(launch_cmd)

    requests_text = requests_file.read_text(encoding="utf-8")
    requests_json: t.List[str] = json.loads(requests_text)

    # verify that there is an item in file for each step added to the batch
    assert len(requests_json) == len(dragon_batch_step.steps)

    for index, req in enumerate(requests_json):
        req_type, req_data = req.split("|", 1)
        # the only steps added are to execute apps, requests should be of type "run"
        assert req_type == "run"

        run_request = DragonRunRequest(**json.loads(req_data))
        assert run_request
        assert run_request.policy.cpu_affinity == cpu_affinities[index]
        assert run_request.policy.gpu_affinity == gpu_affinities[index]
