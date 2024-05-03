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

import pytest

from smartsim._core.config import CONFIG
from smartsim._core.schemas.dragonRequests import *
from smartsim._core.schemas.dragonResponses import *
from smartsim._core.utils.helpers import create_short_id_str

if t.TYPE_CHECKING:
    from smartsim._core.launcher.dragon.dragonBackend import (
        DragonBackend,
        ProcessGroupInfo,
    )


class NodeMock(MagicMock):
    @property
    def hostname(self) -> str:
        return create_short_id_str()


class GroupStateMock(MagicMock):
    def Running(self) -> MagicMock:
        running = MagicMock(**{"__str__.return_value": "Running"})
        return running

    def Error(self) -> MagicMock:
        error = MagicMock(**{"__str__.return_value": "Error"})
        return error


def get_mock_backend(monkeypatch: pytest.MonkeyPatch) -> "DragonBackend":

    process_mock = MagicMock(returncode=0)
    process_module_mock = MagicMock()
    process_module_mock.Process = process_mock
    node_mock = NodeMock()
    system_mock = MagicMock(nodes=["node1", "node2", "node3"])
    monkeypatch.setitem(
        sys.modules,
        "dragon",
        MagicMock(
            **{
                "native.machine.Node.return_value": node_mock,
                "native.machine.System.return_value": system_mock,
                "native.group_state": GroupStateMock(),
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "dragon.infrastructure.connection",
        MagicMock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "dragon.infrastructure.policy",
        MagicMock(**{"Policy.return_value": MagicMock()}),
    )
    monkeypatch.setitem(sys.modules, "dragon.native.process", process_module_mock)
    monkeypatch.setitem(sys.modules, "dragon.native.process_group", MagicMock())

    monkeypatch.setitem(sys.modules, "dragon.native.group_state", GroupStateMock())
    monkeypatch.setitem(
        sys.modules,
        "dragon.native.machine",
        MagicMock(
            **{"System.return_value": system_mock, "Node.return_value": node_mock}
        ),
    )
    from smartsim._core.launcher.dragon.dragonBackend import DragonBackend

    dragon_backend = DragonBackend(pid=99999)
    monkeypatch.setattr(
        dragon_backend, "_free_hosts", collections.deque(dragon_backend._hosts)
    )

    return dragon_backend


def set_mock_group_infos(
    monkeypatch: pytest.MonkeyPatch, dragon_backend: "DragonBackend"
) -> t.Dict[str, "ProcessGroupInfo"]:
    dragon_mock = MagicMock()
    process_mock = MagicMock()
    process_mock.configure_mock(**{"returncode": 0})
    dragon_mock.configure_mock(**{"native.process.Process.return_value": process_mock})
    monkeypatch.setitem(sys.modules, "dragon", dragon_mock)
    from smartsim._core.launcher.dragon.dragonBackend import ProcessGroupInfo

    running_group = MagicMock(status="Running")
    error_group = MagicMock(status="Error")
    hosts = dragon_backend._hosts

    group_infos = {
        "abc123-1": ProcessGroupInfo(
            SmartSimStatus.STATUS_RUNNING,
            running_group,
            [123],
            [],
            hosts[0:1],
            MagicMock(),
        ),
        "del999-2": ProcessGroupInfo(
            SmartSimStatus.STATUS_CANCELLED,
            error_group,
            [124],
            [-9],
            hosts[1:2],
            MagicMock(),
        ),
        "c101vz-3": ProcessGroupInfo(
            SmartSimStatus.STATUS_COMPLETED,
            MagicMock(),
            [125, 126],
            [0],
            hosts[1:3],
            MagicMock(),
        ),
        "0ghjk1-4": ProcessGroupInfo(
            SmartSimStatus.STATUS_FAILED,
            error_group,
            [127],
            [-1],
            hosts[2:3],
            MagicMock(),
        ),
        "ljace0-5": ProcessGroupInfo(
            SmartSimStatus.STATUS_NEVER_STARTED, None, [], [], [], None
        ),
    }

    monkeypatch.setattr(dragon_backend, "_group_infos", group_infos)
    monkeypatch.setattr(dragon_backend, "_free_hosts", collections.deque(hosts[1:3]))
    monkeypatch.setattr(dragon_backend, "_allocated_hosts", {hosts[0]: "abc123-1"})
    monkeypatch.setattr(dragon_backend, "_running_steps", ["abc123-1"])

    return group_infos


def test_handshake_request(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch)

    handshake_req = DragonHandshakeRequest()
    handshake_resp = dragon_backend.process_request(handshake_req)

    assert isinstance(handshake_resp, DragonHandshakeResponse)
    assert handshake_resp.dragon_pid == 99999


def test_run_request(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch)
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

    mock_process_group = MagicMock(puids=[123,124])

    dragon_backend._group_infos[step_id].process_group = mock_process_group
    dragon_backend._group_infos[step_id].puids = [123, 124]
    dragon_backend._start_steps()

    assert dragon_backend._running_steps == [step_id]
    assert len(dragon_backend._queued_steps) == 0
    assert len(dragon_backend._free_hosts) == 1
    assert dragon_backend._allocated_hosts[dragon_backend.hosts[0]] == step_id
    assert dragon_backend._allocated_hosts[dragon_backend.hosts[1]] == step_id

    monkeypatch.setattr(
        dragon_backend._group_infos[step_id].process_group, "status", "Running"
    )

    dragon_backend._update()

    assert dragon_backend._running_steps == [step_id]
    assert len(dragon_backend._queued_steps) == 0
    assert len(dragon_backend._free_hosts) == 1
    assert dragon_backend._allocated_hosts[dragon_backend.hosts[0]] == step_id
    assert dragon_backend._allocated_hosts[dragon_backend.hosts[1]] == step_id

    dragon_backend._group_infos[step_id].status = SmartSimStatus.STATUS_CANCELLED

    dragon_backend._update()
    assert not dragon_backend._running_steps


def test_udpate_status_request(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch)

    group_infos = set_mock_group_infos(monkeypatch, dragon_backend)

    status_update_request = DragonUpdateStatusRequest(step_ids=list(group_infos.keys()))

    status_update_response = dragon_backend.process_request(status_update_request)

    assert isinstance(status_update_response, DragonUpdateStatusResponse)
    assert status_update_response.statuses == {
        step_id: (grp_info.status, grp_info.return_codes)
        for step_id, grp_info in group_infos.items()
    }


def test_stop_request(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch)
    group_infos = set_mock_group_infos(monkeypatch, dragon_backend)

    running_steps = [
        step_id
        for step_id, group in group_infos.items()
        if group.status == SmartSimStatus.STATUS_RUNNING
    ]

    step_id_to_stop = running_steps[0]

    stop_request = DragonStopRequest(step_id=step_id_to_stop)

    stop_response = dragon_backend.process_request(stop_request)

    assert isinstance(stop_response, DragonStopResponse)
    assert len(dragon_backend._stop_requests) == 1

    dragon_backend._update()

    assert len(dragon_backend._stop_requests) == 0
    assert (
        dragon_backend._group_infos[step_id_to_stop].status
        == SmartSimStatus.STATUS_CANCELLED
    )

    assert len(dragon_backend._allocated_hosts) == 0
    assert len(dragon_backend._free_hosts) == 3


@pytest.mark.parametrize(
    "immediate, frontend_shutdown",
    [[True, True], [True, False], [False, True], [False, False]],
)
def test_shutdown_request(
    monkeypatch: pytest.MonkeyPatch, immediate: bool, frontend_shutdown: bool
) -> None:
    monkeypatch.setenv("SMARTSIM_FLAG_TELEMETRY", "0")
    dragon_backend = get_mock_backend(monkeypatch)
    monkeypatch.setattr(dragon_backend, "_cooldown_period", 1)
    set_mock_group_infos(monkeypatch, dragon_backend)

    shutdown_req = DragonShutdownRequest(
        immediate=immediate, frontend_shutdown=frontend_shutdown
    )
    shutdown_resp = dragon_backend.process_request(shutdown_req)

    assert dragon_backend._shutdown_requested
    assert isinstance(shutdown_resp, DragonShutdownResponse)
    assert dragon_backend._can_shutdown == immediate
    assert dragon_backend.frontend_shutdown == frontend_shutdown

    dragon_backend._update()
    assert not dragon_backend.should_shutdown
    time.sleep(dragon_backend._cooldown_period + 0.1)
    dragon_backend._update()

    assert dragon_backend.should_shutdown == immediate
    assert dragon_backend._has_cooled_down == immediate


@pytest.mark.parametrize("telemetry_flag", ["0", "1"])
def test_cooldown_is_set(monkeypatch: pytest.MonkeyPatch, telemetry_flag: str) -> None:
    monkeypatch.setenv("SMARTSIM_FLAG_TELEMETRY", telemetry_flag)
    dragon_backend = get_mock_backend(monkeypatch)

    expected_cooldown = (
        2 * CONFIG.telemetry_frequency + 5 if int(telemetry_flag) > 0 else 5
    )

    if telemetry_flag:
        assert dragon_backend.cooldown_period == expected_cooldown
    else:
        assert dragon_backend.cooldown_period == expected_cooldown


def test_heartbeat_and_time(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch)
    first_heartbeat = dragon_backend.last_heartbeat
    assert dragon_backend.current_time > first_heartbeat
    dragon_backend._heartbeat()
    assert dragon_backend.last_heartbeat > first_heartbeat


@pytest.mark.parametrize("num_nodes", [1, 3, 100])
def test_can_honor(monkeypatch: pytest.MonkeyPatch, num_nodes: int) -> None:
    dragon_backend = get_mock_backend(monkeypatch)
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

    assert dragon_backend._can_honor(run_req)[0] == (
        num_nodes <= len(dragon_backend._hosts)
    )


def test_get_id(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch)
    step_id = next(dragon_backend._step_ids)

    assert step_id.endswith("0")
    assert step_id != next(dragon_backend._step_ids)


def test_view(monkeypatch: pytest.MonkeyPatch) -> None:
    dragon_backend = get_mock_backend(monkeypatch)
    set_mock_group_infos(monkeypatch, dragon_backend)
    hosts = dragon_backend.hosts

    expected_message = textwrap.dedent(f"""\
        Dragon server backend update
        | Host    |  Status  |
        |---------|----------|
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

    assert dragon_backend.status_message == expected_message
