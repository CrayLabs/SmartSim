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

from smartsim._core.launcher.step.dragonStep import DragonBatchStep, DragonStep
from smartsim.settings.dragonRunSettings import DragonRunSettings
from smartsim.settings.slurmSettings import SbatchSettings

try:
    from dragon.infrastructure.policy import Policy

    import smartsim._core.entrypoints.dragon as drg
    from smartsim._core.launcher.dragon.dragonBackend import DragonBackend

    dragon_loaded = True
except:
    dragon_loaded = False

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b

from smartsim._core.schemas.dragonRequests import *
from smartsim._core.schemas.dragonResponses import *


@pytest.fixture
def dragon_batch_step(test_dir: str) -> "DragonBatchStep":
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


@pytest.mark.skipif(not dragon_loaded, reason="Test is only for Dragon WLM systems")
@pytest.mark.parametrize(
    "dragon_request",
    [
        pytest.param(DragonHandshakeRequest(), id="DragonHandshakeRequest"),
        pytest.param(DragonShutdownRequest(), id="DragonShutdownRequest"),
        pytest.param(
            DragonBootstrapRequest(address="localhost"), id="DragonBootstrapRequest"
        ),
    ],
)
def test_create_run_policy_non_run_request(dragon_request: DragonRequest) -> None:
    """Verify that a default policy is returned when a request is
    not attempting to start a new proccess (e.g. a DragonRunRequest)"""
    policy = DragonBackend.create_run_policy(dragon_request, "localhost")

    assert policy is not None, "Default policy was not returned"
    assert policy.cpu_affinity == [], "Default cpu affinity was not empty"
    assert policy.gpu_affinity == [], "Default gpu affinity was not empty"


@pytest.mark.skipif(not dragon_loaded, reason="Test is only for Dragon WLM systems")
def test_create_run_policy_run_request_no_run_policy() -> None:
    """Verify that a policy specifying no policy is returned with all default
    values (no device, empty cpu & gpu affinity)"""
    run_req = DragonRunRequest(
        exe="sleep",
        exe_args=["5"],
        path="/a/fake/path",
        nodes=2,
        tasks=1,
        tasks_per_node=1,
        env={},
        current_env={},
        pmi_enabled=False,
        # policy=  # <--- skipping this
    )

    policy = DragonBackend.create_run_policy(run_req, "localhost")

    assert set(policy.cpu_affinity) == set()
    assert policy.gpu_affinity == []


@pytest.mark.skipif(not dragon_loaded, reason="Test is only for Dragon WLM systems")
def test_create_run_policy_run_request_default_run_policy() -> None:
    """Verify that a policy specifying no affinity is returned with
    default value for device and empty affinity lists"""
    run_req = DragonRunRequest(
        exe="sleep",
        exe_args=["5"],
        path="/a/fake/path",
        nodes=2,
        tasks=1,
        tasks_per_node=1,
        env={},
        current_env={},
        pmi_enabled=False,
        policy=DragonRunPolicy(),  # <--- passing default values
    )

    policy = DragonBackend.create_run_policy(run_req, "localhost")

    assert set(policy.cpu_affinity) == set()
    assert set(policy.gpu_affinity) == set()


@pytest.mark.skipif(not dragon_loaded, reason="Test is only for Dragon WLM systems")
def test_create_run_policy_run_request_cpu_affinity_no_device() -> None:
    """Verify that a input policy specifying a CPU affinity but lacking the device field
    produces a Dragon Policy with the CPU device specified"""
    affinity = set([0, 2, 4])
    run_req = DragonRunRequest(
        exe="sleep",
        exe_args=["5"],
        path="/a/fake/path",
        nodes=2,
        tasks=1,
        tasks_per_node=1,
        env={},
        current_env={},
        pmi_enabled=False,
        policy=DragonRunPolicy(cpu_affinity=list(affinity)),  # <-- no device spec
    )

    policy = DragonBackend.create_run_policy(run_req, "localhost")

    assert set(policy.cpu_affinity) == affinity
    assert policy.gpu_affinity == []


@pytest.mark.skipif(not dragon_loaded, reason="Test is only for Dragon WLM systems")
def test_create_run_policy_run_request_cpu_affinity() -> None:
    """Verify that a policy specifying CPU affinity is returned as expected"""
    affinity = set([0, 2, 4])
    run_req = DragonRunRequest(
        exe="sleep",
        exe_args=["5"],
        path="/a/fake/path",
        nodes=2,
        tasks=1,
        tasks_per_node=1,
        env={},
        current_env={},
        pmi_enabled=False,
        policy=DragonRunPolicy(cpu_affinity=list(affinity)),
    )

    policy = DragonBackend.create_run_policy(run_req, "localhost")

    assert set(policy.cpu_affinity) == affinity
    assert policy.gpu_affinity == []


@pytest.mark.skipif(not dragon_loaded, reason="Test is only for Dragon WLM systems")
def test_create_run_policy_run_request_gpu_affinity() -> None:
    """Verify that a policy specifying GPU affinity is returned as expected"""
    affinity = set([0, 2, 4])
    run_req = DragonRunRequest(
        exe="sleep",
        exe_args=["5"],
        path="/a/fake/path",
        nodes=2,
        tasks=1,
        tasks_per_node=1,
        env={},
        current_env={},
        pmi_enabled=False,
        policy=DragonRunPolicy(device="gpu", gpu_affinity=list(affinity)),
    )

    policy = DragonBackend.create_run_policy(run_req, "localhost")

    assert policy.cpu_affinity == []
    assert set(policy.gpu_affinity) == set(affinity)


@pytest.mark.skipif(not dragon_loaded, reason="Test is only for Dragon WLM systems")
def test_dragon_run_policy_from_run_args() -> None:
    """Verify that a DragonRunPolicy is created from a dictionary of run arguments"""
    run_args = {
        "gpu-affinity": "0,1,2",
        "cpu-affinity": "3,4,5,6",
    }

    policy = DragonRunPolicy.from_run_args(run_args)

    assert policy.cpu_affinity == [3, 4, 5, 6]
    assert policy.gpu_affinity == [0, 1, 2]


def test_dragon_run_policy_from_run_args_empty() -> None:
    """Verify that a DragonRunPolicy is created from an empty
    dictionary of run arguments"""
    run_args = {}

    policy = DragonRunPolicy.from_run_args(run_args)

    assert policy.cpu_affinity == []
    assert policy.gpu_affinity == []


def test_dragon_run_policy_from_run_args_cpu_affinity() -> None:
    """Verify that a DragonRunPolicy is created from a dictionary
    of run arguments containing a CPU affinity"""
    run_args = {
        "cpu-affinity": "3,4,5,6",
    }

    policy = DragonRunPolicy.from_run_args(run_args)

    assert policy.cpu_affinity == [3, 4, 5, 6]
    assert policy.gpu_affinity == []


def test_dragon_run_policy_from_run_args_gpu_affinity() -> None:
    """Verify that a DragonRunPolicy is created from a dictionary
    of run arguments containing a GPU affinity"""
    run_args = {
        "gpu-affinity": "0, 1, 2",
    }

    policy = DragonRunPolicy.from_run_args(run_args)

    assert policy.cpu_affinity == []
    assert policy.gpu_affinity == [0, 1, 2]


def test_dragon_run_policy_from_run_args_invalid_gpu_affinity() -> None:
    """Verify that a DragonRunPolicy is NOT created from a dictionary
    of run arguments with an invalid GPU affinity"""
    run_args = {
        "gpu-affinity": "0,-1,2",
    }

    with pytest.raises(SmartSimError) as ex:
        DragonRunPolicy.from_run_args(run_args)

    assert "DragonRunPolicy" in ex.value.args[0]


def test_dragon_run_policy_from_run_args_invalid_cpu_affinity() -> None:
    """Verify that a DragonRunPolicy is NOT created from a dictionary
    of run arguments with an invalid CPU affinity"""
    run_args = {
        "cpu-affinity": "3,4,5,-6",
    }

    with pytest.raises(SmartSimError) as ex:
        DragonRunPolicy.from_run_args(run_args)

    assert "DragonRunPolicy" in ex.value.args[0]


def test_dragon_run_policy_from_run_args_ignore_empties_gpu() -> None:
    """Verify that a DragonRunPolicy is created from a dictionary
    of run arguments and ignores empty values in the serialized gpu list"""
    run_args = {
        "gpu-affinity": "0,,2",
    }

    policy = DragonRunPolicy.from_run_args(run_args)

    assert policy.cpu_affinity == []
    assert policy.gpu_affinity == [0, 2]


def test_dragon_run_policy_from_run_args_ignore_empties_cpu() -> None:
    """Verify that a DragonRunPolicy is created from a dictionary
    of run arguments and ignores empty values in the serialized cpu list"""
    run_args = {
        "cpu-affinity": "3,4,,6,",
    }

    policy = DragonRunPolicy.from_run_args(run_args)

    assert policy.cpu_affinity == [3, 4, 6]
    assert policy.gpu_affinity == []


def test_dragon_run_policy_from_run_args_null_gpu_affinity() -> None:
    """Verify that a DragonRunPolicy is created if a null value is encountered
    in the gpu-affinity list"""
    run_args = {
        "gpu-affinity": None,
        "cpu-affinity": "3,4,5,6",
    }

    policy = DragonRunPolicy.from_run_args(run_args)

    assert policy.cpu_affinity == [3, 4, 5, 6]
    assert policy.gpu_affinity == []


def test_dragon_run_policy_from_run_args_null_cpu_affinity() -> None:
    """Verify that a DragonRunPolicy is created if a null value is encountered
    in the cpu-affinity list"""
    run_args = {"gpu-affinity": "0,1,2", "cpu-affinity": None}

    policy = DragonRunPolicy.from_run_args(run_args)

    assert policy.cpu_affinity == []
    assert policy.gpu_affinity == [0, 1, 2]
