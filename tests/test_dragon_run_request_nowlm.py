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
from pydantic import ValidationError

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a

from smartsim._core.schemas.dragonRequests import *
from smartsim._core.schemas.dragonResponses import *


def test_run_request_with_null_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that an empty policy does not cause an error"""
    # dragon_backend = get_mock_backend(monkeypatch)
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


def test_run_request_with_empty_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a non-empty policy is set correctly"""
    # dragon_backend = get_mock_backend(monkeypatch)
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
        policy=DragonRunPolicy(),
    )
    assert run_req.policy is not None
    assert not run_req.policy.cpu_affinity
    assert not run_req.policy.gpu_affinity


@pytest.mark.parametrize(
    "device,cpu_affinity,gpu_affinity",
    [
        pytest.param("cpu", [-1], [], id="cpu_affinity"),
        pytest.param("gpu", [], [-1], id="gpu_affinity"),
    ],
)
def test_run_request_with_negative_affinity(
    device: str,
    cpu_affinity: t.List[int],
    gpu_affinity: t.List[int],
) -> None:
    """Verify that invalid affinity values fail validation"""
    with pytest.raises(ValidationError) as ex:
        DragonRunRequest(
            exe="sleep",
            exe_args=["5"],
            path="/a/fake/path",
            nodes=2,
            tasks=1,
            tasks_per_node=1,
            env={},
            current_env={},
            pmi_enabled=False,
            policy=DragonRunPolicy(
                cpu_affinity=cpu_affinity, gpu_affinity=gpu_affinity
            ),
        )

    assert f"{device}_affinity" in str(ex.value)
    assert "greater than or equal to 0" in str(ex.value)
