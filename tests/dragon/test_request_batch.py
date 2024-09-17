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

from smartsim._core.mli.infrastructure.storage.feature_store import FeatureStoreKey
from smartsim._core.mli.infrastructure.worker.worker import (
    InferenceRequest,
    RequestBatch,
    TransformInputResult,
)
from smartsim._core.mli.message_handler import MessageHandler

handler = MessageHandler()


@pytest.fixture
def request_batch() -> RequestBatch:
    return RequestBatch([InferenceRequest()], None, fs_key)


@pytest.fixture
def fs_key() -> FeatureStoreKey:
    return FeatureStoreKey("key", "descriptor")


@pytest.mark.parametrize(
    "raw_model, expected",
    [
        (handler.build_model(b"bytes", "Model Name", "V1"), True),
        (None, False),
    ],
)
def test_has_raw_model(monkeypatch, request_batch, raw_model, expected):
    """Test the has_raw_model property with different values for raw_model."""
    monkeypatch.setattr(request_batch.requests[0], "raw_model", raw_model)
    assert request_batch.has_raw_model == expected


@pytest.mark.parametrize(
    "raw_model, expected",
    [
        (handler.build_model(b"bytes", "Model Name", "V1"), True),
        (None, False),
    ],
)
def test_has_raw_model_data(monkeypatch, request_batch, raw_model, expected):
    """Test the has_raw_model_data property with different values for raw_model."""
    monkeypatch.setattr(request_batch.requests[0], "raw_model", raw_model)
    assert request_batch.has_raw_model == expected


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ([TransformInputResult(b"results", [], [[1, 2]], ["float"])], True),
        (None, False),
        ([], False),
    ],
)
def test_has_inputs(monkeypatch, request_batch, inputs, expected):
    """Test the has_inputs property with different values for inputs."""
    monkeypatch.setattr(request_batch, "inputs", inputs)
    assert request_batch.has_inputs == expected
