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

import typing as t

import pytest

dragon = pytest.importorskip("dragon")

from smartsim._core.mli.infrastructure.control.worker_manager import build_failure_reply

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


@pytest.mark.parametrize(
    "status, message",
    [
        pytest.param("timeout", "Worker timed out", id="timeout"),
        pytest.param("fail", "Failed while executing", id="fail"),
    ],
)
def test_build_failure_reply(status: "Status", message: str):
    "Ensures failure replies can be built successfully"
    response = build_failure_reply(status, message)
    display_name = response.schema.node.displayName  # type: ignore
    class_name = display_name.split(":")[-1]
    assert class_name == "Response"
    assert response.status == status
    assert response.message == message


def test_build_failure_reply_fails():
    "Ensures ValueError is raised if a Status Enum is not used"
    with pytest.raises(ValueError) as ex:
        response = build_failure_reply("not a status enum", "message")

    assert "Error assigning status to response" in ex.value.args[0]
