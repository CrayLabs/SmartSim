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
from smartsim._core.mli.infrastructure.worker.worker import InferenceRequest
from smartsim._core.mli.message_handler import MessageHandler

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon

handler = MessageHandler()


@pytest.fixture
def inference_request() -> InferenceRequest:
    return InferenceRequest()


@pytest.fixture
def fs_key() -> TensorKey:
    return TensorKey("key", "descriptor")


@pytest.mark.parametrize(
    "raw_model, expected",
    [
        (handler.build_model(b"bytes", "Model Name", "V1"), True),
        (None, False),
    ],
)
def test_has_raw_model(monkeypatch, inference_request, raw_model, expected):
    """Test the has_raw_model property with different values for raw_model."""
    monkeypatch.setattr(inference_request, "raw_model", raw_model)
    assert inference_request.has_raw_model == expected


@pytest.mark.parametrize(
    "model_key, expected",
    [
        (fs_key, True),
        (None, False),
    ],
)
def test_has_model_key(monkeypatch, inference_request, model_key, expected):
    """Test the has_model_key property with different values for model_key."""
    monkeypatch.setattr(inference_request, "model_key", model_key)
    assert inference_request.has_model_key == expected


@pytest.mark.parametrize(
    "raw_inputs, expected",
    [([b"raw input bytes"], True), (None, False), ([], False)],
)
def test_has_raw_inputs(monkeypatch, inference_request, raw_inputs, expected):
    """Test the has_raw_inputs property with different values for raw_inputs."""
    monkeypatch.setattr(inference_request, "raw_inputs", raw_inputs)
    assert inference_request.has_raw_inputs == expected


@pytest.mark.parametrize(
    "input_keys, expected",
    [([fs_key], True), (None, False), ([], False)],
)
def test_has_input_keys(monkeypatch, inference_request, input_keys, expected):
    """Test the has_input_keys property with different values for input_keys."""
    monkeypatch.setattr(inference_request, "input_keys", input_keys)
    assert inference_request.has_input_keys == expected


@pytest.mark.parametrize(
    "output_keys, expected",
    [([fs_key], True), (None, False), ([], False)],
)
def test_has_output_keys(monkeypatch, inference_request, output_keys, expected):
    """Test the has_output_keys property with different values for output_keys."""
    monkeypatch.setattr(inference_request, "output_keys", output_keys)
    assert inference_request.has_output_keys == expected


@pytest.mark.parametrize(
    "input_meta, expected",
    [
        ([handler.build_tensor_descriptor("c", "float32", [1, 2, 3])], True),
        (None, False),
        ([], False),
    ],
)
def test_has_input_meta(monkeypatch, inference_request, input_meta, expected):
    """Test the has_input_meta property with different values for input_meta."""
    monkeypatch.setattr(inference_request, "input_meta", input_meta)
    assert inference_request.has_input_meta == expected
