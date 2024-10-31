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

import collections
import sys
import textwrap
import time
from unittest.mock import MagicMock

import pydantic.error_wrappers
import pytest

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b
dragon = pytest.importorskip("dragon")

from smartsim._core.config import CONFIG
from smartsim._core.launcher.dragon.dragon_backend import (
    DragonBackend,
    ProcessGroupInfo,
)
from smartsim._core.launcher.dragon.pqueue import NodePrioritizer
from smartsim._core.schemas.dragon_requests import *
from smartsim._core.schemas.dragon_responses import *
from smartsim.status import TERMINAL_STATUSES, InvalidJobStatus, JobStatus


class GroupStateMock(MagicMock):
    def Running(self) -> MagicMock:
        running = MagicMock(**{"__str__.return_value": "Running"})
        return running

    def Error(self) -> MagicMock:
        error = MagicMock(**{"__str__.return_value": "Error"})
        return error


class ProcessGroupMock(MagicMock):
    puids = [121, 122]


def get_mock_backend(
    monkeypatch: pytest.MonkeyPatch, num_cpus: int, num_gpus: int
) -> "DragonBackend":
    # create all the necessary namespaces as raw magic mocks
    monkeypatch.setitem(sys.modules, "dragon.data.ddict.ddict", MagicMock())
    monkeypatch.setitem(sys.modules, "dragon.native.machine", MagicMock())
    monkeypatch.setitem(sys.modules, "dragon.native.group_state", MagicMock())
    monkeypatch.setitem(sys.modules, "dragon.native.process_group", MagicMock())
    monkeypatch.setitem(sys.modules, "dragon.native.process", MagicMock())
    monkeypatch.setitem(sys.modules, "dragon.infrastructure.connection", MagicMock())
    monkeypatch.setitem(sys.modules, "dragon.infrastructure.policy", MagicMock())
    monkeypatch.setitem(sys.modules, "dragon.infrastructure.process_desc", MagicMock())
    monkeypatch.setitem(sys.modules, "dragon.data.ddict.ddict", MagicMock())

    node_list = ["node1", "node2", "node3"]
    system_mock = MagicMock(return_value=MagicMock(nodes=node_list))
    node_mock = lambda x: MagicMock(hostname=x, num_cpus=num_cpus, num_gpus=num_gpus)
    process_group_mock = MagicMock(return_value=ProcessGroupMock())
    process_mock = MagicMock(returncode=0)
    policy_mock = MagicMock(return_value=MagicMock())
    group_state_mock = GroupStateMock()

    # customize members that must perform specific actions within the namespaces
    monkeypatch.setitem(
        sys.modules,
        "dragon",
        MagicMock(
            **{
                "native.machine.Node": node_mock,
                "native.machine.System": system_mock,
                "native.group_state": group_state_mock,
                "native.process_group.ProcessGroup": process_group_mock,
                "native.process_group.Process": process_mock,
                "native.process.Process": process_mock,
                "infrastructure.policy.Policy": policy_mock,
            }
        ),
    )

    dragon_backend = DragonBackend(pid=99999)

    # NOTE: we're manually updating these values due to issue w/mocking namespaces
    dragon_backend._prioritizer = NodePrioritizer(
        [
            MagicMock(num_cpus=num_cpus, num_gpus=num_gpus, hostname=node)
            for node in node_list
        ],
        dragon_backend._queue_lock,
    )
    dragon_backend._cpus = [num_cpus] * len(node_list)
    dragon_backend._gpus = [num_gpus] * len(node_list)

    return dragon_backend


def set_mock_group_infos(
    monkeypatch: pytest.MonkeyPatch, dragon_backend: "DragonBackend"
) -> t.Dict[str, "ProcessGroupInfo"]:
    dragon_mock = MagicMock()
    process_mock = MagicMock()
    process_mock.configure_mock(**{"returncode": 0})
    dragon_mock.configure_mock(**{"native.process.Process.return_value": process_mock})
    monkeypatch.setitem(sys.modules, "dragon", dragon_mock)
    from smartsim._core.launcher.dragon.dragon_backend import ProcessGroupInfo

    running_group = MagicMock(status="Running")
    error_group = MagicMock(status="Error")
    hosts = dragon_backend._hosts

    group_infos = {
        "abc123-1": ProcessGroupInfo(
            JobStatus.RUNNING,
            running_group,
            [123],
            [],
            hosts[0:1],
            MagicMock(),
        ),
        "del999-2": ProcessGroupInfo(
            JobStatus.CANCELLED,
            error_group,
            [124],
            [-9],
            hosts[1:2],
            MagicMock(),
        ),
        "c101vz-3": ProcessGroupInfo(
            JobStatus.COMPLETED,
            MagicMock(),
            [125, 126],
            [0],
            hosts[1:3],
            MagicMock(),
        ),
        "0ghjk1-4": ProcessGroupInfo(
            JobStatus.FAILED,
            error_group,
            [127],
            [-1],
            hosts[2:3],
            MagicMock(),
        ),
        "ljace0-5": ProcessGroupInfo(
            InvalidJobStatus.NEVER_STARTED, None, [], [], [], None
        ),
    }

    monkeypatch.setattr(dragon_backend, "_group_infos", group_infos)
    monkeypatch.setattr(dragon_backend, "_allocated_hosts", {hosts[0]: {"abc123-1"}})
    monkeypatch.setattr(dragon_backend, "_running_steps", ["abc123-1"])

    return group_infos


def test_handshake_request(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)

    handshake_req = DragonHandshakeRequest()
    handshake_resp = dragon_backend.process_request(handshake_req)

    assert isinstance(handshake_resp, DragonHandshakeResponse)
    assert handshake_resp.dragon_pid == 99999


def test_run_request(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)
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
    )

    run_resp = dragon_backend.process_request(run_req)
    assert isinstance(run_resp, DragonRunResponse)

    step_id = run_resp.step_id
    assert dragon_backend._queued_steps[step_id] == run_req

    mock_process_group = MagicMock(puids=[123, 124])

    dragon_backend._group_infos[step_id].process_group = mock_process_group
    dragon_backend._group_infos[step_id].puids = [123, 124]
    dragon_backend._start_steps()

    assert dragon_backend._running_steps == [step_id]
    assert len(dragon_backend._queued_steps) == 0
    assert len(dragon_backend.free_hosts) == 1
    assert step_id in dragon_backend._allocated_hosts[dragon_backend.hosts[0]]
    assert step_id in dragon_backend._allocated_hosts[dragon_backend.hosts[1]]

    monkeypatch.setattr(
        dragon_backend._group_infos[step_id].process_group, "status", "Running"
    )

    dragon_backend._update()

    assert dragon_backend._running_steps == [step_id]
    assert len(dragon_backend._queued_steps) == 0
    assert len(dragon_backend.free_hosts) == 1
    assert step_id in dragon_backend._allocated_hosts[dragon_backend.hosts[0]]
    assert step_id in dragon_backend._allocated_hosts[dragon_backend.hosts[1]]

    dragon_backend._group_infos[step_id].status = JobStatus.CANCELLED

    dragon_backend._update()
    assert not dragon_backend._running_steps


def test_deny_run_request(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)

    dragon_backend._shutdown_requested = True

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
    )

    run_resp = dragon_backend.process_request(run_req)
    assert isinstance(run_resp, DragonRunResponse)
    assert run_resp.error_message == "Cannot satisfy request, server is shutting down."
    step_id = run_resp.step_id

    assert dragon_backend.group_infos[step_id].status == JobStatus.FAILED


def test_run_request_with_empty_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a policy is applied to a run request"""
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)
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
        policy=None,
    )
    assert run_req.policy is None


def test_run_request_with_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a policy is applied to a run request"""
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)
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
        policy=DragonRunPolicy(cpu_affinity=[0, 1]),
    )

    run_resp = dragon_backend.process_request(run_req)
    assert isinstance(run_resp, DragonRunResponse)

    step_id = run_resp.step_id
    assert dragon_backend._queued_steps[step_id] == run_req

    mock_process_group = MagicMock(puids=[123, 124])

    dragon_backend._group_infos[step_id].process_group = mock_process_group
    dragon_backend._group_infos[step_id].puids = [123, 124]
    dragon_backend._start_steps()

    assert dragon_backend._running_steps == [step_id]
    assert len(dragon_backend._queued_steps) == 0
    assert len(dragon_backend._prioritizer.unassigned()) == 1
    assert step_id in dragon_backend._allocated_hosts[dragon_backend.hosts[0]]
    assert step_id in dragon_backend._allocated_hosts[dragon_backend.hosts[1]]

    monkeypatch.setattr(
        dragon_backend._group_infos[step_id].process_group, "status", "Running"
    )

    dragon_backend._update()

    assert dragon_backend._running_steps == [step_id]
    assert len(dragon_backend._queued_steps) == 0
    assert len(dragon_backend._prioritizer.unassigned()) == 1
    assert step_id in dragon_backend._allocated_hosts[dragon_backend.hosts[0]]
    assert step_id in dragon_backend._allocated_hosts[dragon_backend.hosts[1]]

    dragon_backend._group_infos[step_id].status = JobStatus.CANCELLED

    dragon_backend._update()
    assert not dragon_backend._running_steps


def test_udpate_status_request(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)

    group_infos = set_mock_group_infos(monkeypatch, dragon_backend)

    status_update_request = DragonUpdateStatusRequest(step_ids=list(group_infos.keys()))

    status_update_response = dragon_backend.process_request(status_update_request)

    assert isinstance(status_update_response, DragonUpdateStatusResponse)
    assert status_update_response.statuses == {
        step_id: (grp_info.status, grp_info.return_codes)
        for step_id, grp_info in group_infos.items()
    }


def test_stop_request(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)
    group_infos = set_mock_group_infos(monkeypatch, dragon_backend)

    running_steps = [
        step_id
        for step_id, group in group_infos.items()
        if group.status == JobStatus.RUNNING
    ]

    step_id_to_stop = running_steps[0]

    stop_request = DragonStopRequest(step_id=step_id_to_stop)

    stop_response = dragon_backend.process_request(stop_request)

    assert isinstance(stop_response, DragonStopResponse)
    assert len(dragon_backend._stop_requests) == 1

    dragon_backend._update()

    assert len(dragon_backend._stop_requests) == 0
    assert dragon_backend._group_infos[step_id_to_stop].status == JobStatus.CANCELLED

    assert len(dragon_backend._allocated_hosts) == 0
    assert len(dragon_backend._prioritizer.unassigned()) == 3


@pytest.mark.parametrize(
    "immediate, kill_jobs, frontend_shutdown",
    [
        [True, True, True],
        [True, True, False],
        [True, False, True],
        [True, False, False],
        [False, True, True],
        [False, True, False],
    ],
)
def test_shutdown_request(
    monkeypatch: pytest.MonkeyPatch,
    immediate: bool,
    kill_jobs: bool,
    frontend_shutdown: bool,
) -> None:
    monkeypatch.setenv("SMARTSIM_FLAG_TELEMETRY", "0")
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)
    monkeypatch.setattr(dragon_backend, "_cooldown_period", 1)
    set_mock_group_infos(monkeypatch, dragon_backend)

    if kill_jobs:
        for group_info in dragon_backend.group_infos.values():
            if not group_info.status in TERMINAL_STATUSES:
                group_info.status = JobStatus.FAILED
                group_info.return_codes = [-9]
            group_info.process_group = None
            group_info.redir_workers = None
        dragon_backend._running_steps.clear()

    shutdown_req = DragonShutdownRequest(
        immediate=immediate, frontend_shutdown=frontend_shutdown
    )
    shutdown_resp = dragon_backend.process_request(shutdown_req)

    if not kill_jobs:
        stop_request_ids = (
            stop_request.step_id for stop_request in dragon_backend._stop_requests
        )
        for step_id, group_info in dragon_backend.group_infos.items():
            if not group_info.status in TERMINAL_STATUSES:
                assert step_id in stop_request_ids

    assert isinstance(shutdown_resp, DragonShutdownResponse)
    assert dragon_backend._shutdown_requested
    assert dragon_backend.frontend_shutdown == frontend_shutdown

    dragon_backend._update()
    assert not dragon_backend.should_shutdown
    time.sleep(dragon_backend._cooldown_period + 0.1)
    dragon_backend._update()

    assert dragon_backend._can_shutdown == kill_jobs
    assert dragon_backend.should_shutdown == kill_jobs
    assert dragon_backend._has_cooled_down == kill_jobs


@pytest.mark.parametrize("telemetry_flag", ["0", "1"])
def test_cooldown_is_set(monkeypatch: pytest.MonkeyPatch, telemetry_flag: str) -> None:
    monkeypatch.setenv("SMARTSIM_FLAG_TELEMETRY", telemetry_flag)
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)

    expected_cooldown = (
        2 * CONFIG.telemetry_frequency + 5 if int(telemetry_flag) > 0 else 5
    )

    if telemetry_flag:
        assert dragon_backend.cooldown_period == expected_cooldown
    else:
        assert dragon_backend.cooldown_period == expected_cooldown


def test_heartbeat_and_time(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)
    first_heartbeat = dragon_backend.last_heartbeat
    assert dragon_backend.current_time > first_heartbeat
    dragon_backend._heartbeat()
    assert dragon_backend.last_heartbeat > first_heartbeat


@pytest.mark.parametrize("num_nodes", [1, 3, 100])
def test_can_honor(monkeypatch: pytest.MonkeyPatch, num_nodes: int) -> None:
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)
    run_req = DragonRunRequest(
        exe="sleep",
        exe_args=["5"],
        path="/a/fake/path",
        nodes=num_nodes,
        tasks=1,
        tasks_per_node=1,
        env={},
        current_env={},
        pmi_enabled=False,
    )

    can_honor, error_msg = dragon_backend._can_honor(run_req)

    nodes_in_range = num_nodes <= len(dragon_backend._hosts)
    assert can_honor == nodes_in_range
    assert error_msg is None if nodes_in_range else error_msg is not None


@pytest.mark.parametrize("num_nodes", [-10, -1, 0])
def test_can_honor_invalid_num_nodes(
    monkeypatch: pytest.MonkeyPatch, num_nodes: int
) -> None:
    """Verify that requests for invalid numbers of nodes (negative, zero) are rejected"""
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)

    with pytest.raises(pydantic.error_wrappers.ValidationError) as ex:
        DragonRunRequest(
            exe="sleep",
            exe_args=["5"],
            path="/a/fake/path",
            nodes=num_nodes,
            tasks=1,
            tasks_per_node=1,
            env={},
            current_env={},
            pmi_enabled=False,
        )


@pytest.mark.parametrize("affinity", [[0], [0, 1], list(range(8))])
def test_can_honor_cpu_affinity(
    monkeypatch: pytest.MonkeyPatch, affinity: t.List[int]
) -> None:
    """Verify that valid CPU affinities are accepted"""
    num_cpus, num_gpus = 8, 0
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=num_cpus, num_gpus=num_gpus)

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
        policy=DragonRunPolicy(cpu_affinity=affinity),
    )

    assert dragon_backend._can_honor(run_req)[0]


def test_can_honor_cpu_affinity_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that invalid CPU affinities are NOT accepted
    NOTE: negative values are captured by the Pydantic schema"""
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)
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
        policy=DragonRunPolicy(cpu_affinity=list(range(9))),
    )

    assert not dragon_backend._can_honor(run_req)[0]


@pytest.mark.parametrize("affinity", [[0], [0, 1]])
def test_can_honor_gpu_affinity(
    monkeypatch: pytest.MonkeyPatch, affinity: t.List[int]
) -> None:
    """Verify that valid GPU affinities are accepted"""

    num_cpus, num_gpus = 8, 2
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=num_cpus, num_gpus=num_gpus)

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
        policy=DragonRunPolicy(gpu_affinity=affinity),
    )

    assert dragon_backend._can_honor(run_req)[0]


def test_can_honor_gpu_affinity_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that invalid GPU affinities are NOT accepted
    NOTE: negative values are captured by the Pydantic schema"""
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)
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
        policy=DragonRunPolicy(gpu_affinity=list(range(3))),
    )

    assert not dragon_backend._can_honor(run_req)[0]


def test_can_honor_gpu_device_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a request for a GPU if none exists is not accepted"""

    # create a mock node class that always reports no GPUs available
    with monkeypatch.context() as ctx:
        dragon_backend = get_mock_backend(ctx, num_cpus=8, num_gpus=0)

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
            # specify GPU device w/no affinity
            policy=DragonRunPolicy(gpu_affinity=[0]),
        )
        can_honor, _ = dragon_backend._can_honor(run_req)
        assert not can_honor


def test_get_id(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)
    step_id = next(dragon_backend._step_ids)

    assert step_id.endswith("0")
    assert step_id != next(dragon_backend._step_ids)


def test_view(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)
    set_mock_group_infos(monkeypatch, dragon_backend)
    hosts = dragon_backend.hosts
    dragon_backend._prioritizer.increment(hosts[0])

    expected_msg = textwrap.dedent(f"""\
        Dragon server backend update
        | Host   |  Status  |
        |--------|----------|
        | {hosts[0]} |   Busy   |
        | {hosts[1]} |   Free   |
        | {hosts[2]} |   Free   |
        | Step     | Status       | Hosts           |  Return codes  |  Num procs  |
        |----------|--------------|-----------------|----------------|-------------|
        | abc123-1 | Running      | {hosts[0]}         |                |      1      |
        | del999-2 | Cancelled    | {hosts[1]}         |       -9       |      1      |
        | c101vz-3 | Completed    | {hosts[1]},{hosts[2]} |       0        |      2      |
        | 0ghjk1-4 | Failed       | {hosts[2]}         |       -1       |      1      |
        | ljace0-5 | NeverStarted |                 |                |      0      |""")

    # get rid of white space to make the comparison easier
    actual_msg = dragon_backend.status_message.replace(" ", "")
    expected_msg = expected_msg.replace(" ", "")

    # ignore dashes in separators (hostname changes may cause column expansion)
    while actual_msg.find("--") > -1:
        actual_msg = actual_msg.replace("--", "-")
    while expected_msg.find("--") > -1:
        expected_msg = expected_msg.replace("--", "-")

    assert actual_msg == expected_msg


def test_can_honor_hosts_unavailable_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that requesting nodes with invalid names causes number of available
    nodes check to fail due to valid # of named nodes being under num_nodes"""
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)

    # let's supply 2 invalid and 1 valid hostname
    actual_hosts = list(dragon_backend._hosts)
    actual_hosts[0] = f"x{actual_hosts[0]}"
    actual_hosts[1] = f"x{actual_hosts[1]}"

    host_list = ",".join(actual_hosts)

    run_req = DragonRunRequest(
        exe="sleep",
        exe_args=["5"],
        path="/a/fake/path",
        nodes=2,  # <----- requesting 2 of 3 available nodes
        hostlist=host_list,  # <--- only one valid name available
        tasks=1,
        tasks_per_node=1,
        env={},
        current_env={},
        pmi_enabled=False,
        policy=DragonRunPolicy(),
    )

    can_honor, error_msg = dragon_backend._can_honor(run_req)

    # confirm the failure is indicated
    assert not can_honor
    # confirm failure message indicates number of nodes requested as cause
    assert "named hosts" in error_msg


def test_can_honor_hosts_unavailable_hosts_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that requesting nodes with invalid names causes number of available
    nodes check to be reduced but still passes if enough valid named nodes are passed"""
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)

    # let's supply 2 valid and 1 invalid hostname
    actual_hosts = list(dragon_backend._hosts)
    actual_hosts[0] = f"x{actual_hosts[0]}"

    host_list = ",".join(actual_hosts)

    run_req = DragonRunRequest(
        exe="sleep",
        exe_args=["5"],
        path="/a/fake/path",
        nodes=2,  # <----- requesting 2 of 3 available nodes
        hostlist=host_list,  # <--- two valid names are available
        tasks=1,
        tasks_per_node=1,
        env={},
        current_env={},
        pmi_enabled=False,
        policy=DragonRunPolicy(),
    )

    can_honor, error_msg = dragon_backend._can_honor(run_req)

    # confirm the failure is indicated
    assert can_honor, error_msg
    # confirm failure message indicates number of nodes requested as cause
    assert error_msg is None, error_msg


def test_can_honor_hosts_1_hosts_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that requesting nodes with invalid names causes number of available
    nodes check to be reduced but still passes if enough valid named nodes are passed"""
    dragon_backend = get_mock_backend(monkeypatch, num_cpus=8, num_gpus=0)

    # let's supply 2 valid and 1 invalid hostname
    actual_hosts = list(dragon_backend._hosts)
    actual_hosts[0] = f"x{actual_hosts[0]}"

    host_list = ",".join(actual_hosts)

    run_req = DragonRunRequest(
        exe="sleep",
        exe_args=["5"],
        path="/a/fake/path",
        nodes=1,  # <----- requesting 0 nodes - should be ignored
        hostlist=host_list,  # <--- two valid names are available
        tasks=1,
        tasks_per_node=1,
        env={},
        current_env={},
        pmi_enabled=False,
        policy=DragonRunPolicy(),
    )

    can_honor, error_msg = dragon_backend._can_honor(run_req)

    # confirm the failure is indicated
    assert can_honor, error_msg
