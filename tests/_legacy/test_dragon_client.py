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
import typing as t
from unittest.mock import MagicMock

import pytest

from smartsim._core.launcher.step.dragon_step import DragonBatchStep, DragonStep
from smartsim.settings import DragonRunSettings
from smartsim.settings.slurmSettings import SbatchSettings

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


import smartsim._core.entrypoints.dragon_client as dragon_client
from smartsim._core.schemas.dragon_requests import *
from smartsim._core.schemas.dragon_responses import *


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


def get_request_path_from_batch_script(launch_cmd: t.List[str]) -> pathlib.Path:
    """Helper method for finding the path to a request file from the launch command"""
    script_path = pathlib.Path(launch_cmd[-1])
    batch_script = script_path.read_text(encoding="utf-8")
    batch_statements = [line for line in batch_script.split("\n") if line]
    entrypoint_cmd = batch_statements[-1]
    requests_file = pathlib.Path(entrypoint_cmd.split()[-1])
    return requests_file


def test_dragon_client_main_no_arg(monkeypatch: pytest.MonkeyPatch):
    """Verify the client fails when the path to a submission file is not provided."""
    with pytest.raises(SystemExit):
        dragon_client.cleanup = MagicMock()
        dragon_client.main([])

    # arg parser failures occur before resource allocation and should
    # not result in resource cleanup being called
    assert not dragon_client.cleanup.called


def test_dragon_client_main_empty_arg(test_dir: str):
    """Verify the client fails when the path to a submission file is empty."""

    with pytest.raises(ValueError) as ex:
        dragon_client.cleanup = MagicMock()
        dragon_client.main(["+submit", ""])

    # verify it's a value error related to submit argument
    assert "file not provided" in ex.value.args[0]

    # arg parser failures occur before resource allocation and should
    # not result in resource cleanup being called
    assert not dragon_client.cleanup.called


def test_dragon_client_main_bad_arg(test_dir: str):
    """Verify the client returns a failure code when the path to a submission file is
    invalid and does not raise an exception"""
    path = pathlib.Path(test_dir) / "nonexistent_file.json"

    dragon_client.cleanup = MagicMock()
    return_code = dragon_client.main(["+submit", str(path)])

    # ensure non-zero return code
    assert return_code != 0

    # ensure failures do not block resource cleanup
    assert dragon_client.cleanup.called


def test_dragon_client_main(
    dragon_batch_step: DragonBatchStep, monkeypatch: pytest.MonkeyPatch
):
    """Verify the client returns a failure code when the path to a submission file is
    invalid and does not raise an exception"""
    launch_cmd = dragon_batch_step.get_launch_cmd()
    path = get_request_path_from_batch_script(launch_cmd)
    num_requests_in_batch = 4
    num_shutdown_requests = 1
    request_count = num_requests_in_batch + num_shutdown_requests
    submit_value = str(path)

    mock_connector = MagicMock()  # DragonConnector
    mock_connector.is_connected = True
    mock_connector.send_request.return_value = DragonRunResponse(step_id="mock_step_id")
    # mock can_monitor to exit before the infinite loop checking for shutdown
    mock_connector.can_monitor = False

    mock_connector_class = MagicMock()
    mock_connector_class.return_value = mock_connector

    # with monkeypatch.context() as ctx:
    dragon_client.DragonConnector = mock_connector_class
    dragon_client.cleanup = MagicMock()

    return_code = dragon_client.main(["+submit", submit_value])

    # verify each request in the request file was processed
    assert mock_connector.send_request.call_count == request_count

    # we know the batch fixture has a step with no affinity args supplied. skip it
    for i in range(1, num_requests_in_batch):
        sent_args = mock_connector.send_request.call_args_list[i][0]
        request_arg = sent_args[0]

        assert isinstance(request_arg, DragonRunRequest)

        policy = request_arg.policy

        # make sure each policy has been read in correctly with valid affinity indices
        assert len(policy.cpu_affinity) == len(set(policy.cpu_affinity))
        assert len(policy.gpu_affinity) == len(set(policy.gpu_affinity))

    # we get a non-zero due to avoiding the infinite loop. consider refactoring
    assert return_code == os.EX_IOERR

    # ensure failures do not block resource cleanup
    assert dragon_client.cleanup.called
