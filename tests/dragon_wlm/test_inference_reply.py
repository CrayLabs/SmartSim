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

dragon = pytest.importorskip("dragon")

from smartsim._core.mli.infrastructure.storage.feature_store import TensorKey
from smartsim._core.mli.infrastructure.worker.worker import InferenceReply
from smartsim._core.mli.message_handler import MessageHandler

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon

handler = MessageHandler()


@pytest.fixture
def inference_reply() -> InferenceReply:
    return InferenceReply()


@pytest.fixture
def fs_key() -> TensorKey:
    return TensorKey("key", "descriptor")


@pytest.mark.parametrize(
    "outputs, expected",
    [
        ([b"output bytes"], True),
        (None, False),
        ([], False),
    ],
)
def test_has_outputs(monkeypatch, inference_reply, outputs, expected):
    """Test the has_outputs property with different values for outputs."""
    monkeypatch.setattr(inference_reply, "outputs", outputs)
    assert inference_reply.has_outputs == expected


@pytest.mark.parametrize(
    "output_keys, expected",
    [
        ([fs_key], True),
        (None, False),
        ([], False),
    ],
)
def test_has_output_keys(monkeypatch, inference_reply, output_keys, expected):
    """Test the has_output_keys property with different values for output_keys."""
    monkeypatch.setattr(inference_reply, "output_keys", output_keys)
    assert inference_reply.has_output_keys == expected
