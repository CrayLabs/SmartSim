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

import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStoreKey
from smartsim._core.mli.infrastructure.worker.torch_worker import TorchWorker
from smartsim._core.mli.infrastructure.worker.worker import (
    ExecuteResult,
    FetchInputResult,
    FetchModelResult,
    InferenceRequest,
    LoadModelResult,
    TransformInputResult,
)
from smartsim._core.mli.message_handler import MessageHandler
from smartsim.log import get_logger

logger = get_logger(__name__)
# The tests in this file belong to the group_dragon group
pytestmark = pytest.mark.group_dragon


# simple MNIST in PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


torch_device = {"cpu": "cpu", "gpu": "cuda"}


def get_batch() -> torch.Tensor:
    return torch.rand(20, 1, 28, 28)


def create_torch_model():
    n = Net()
    example_forward_input = get_batch()
    module = torch.jit.trace(n, example_forward_input)
    model_buffer = io.BytesIO()
    torch.jit.save(module, model_buffer)
    return model_buffer.getvalue()


def get_request() -> InferenceRequest:

    tensors = [get_batch() for _ in range(2)]
    tensor_numpy = [tensor.numpy() for tensor in tensors]
    serialized_tensors_descriptors = [
        MessageHandler.build_tensor_descriptor("c", "float32", list(tensor.shape))
        for tensor in tensors
    ]

    return InferenceRequest(
        model_key=FeatureStoreKey(key="model", descriptor="xyz"),
        callback=None,
        raw_inputs=tensor_numpy,
        input_keys=None,
        input_meta=serialized_tensors_descriptors,
        output_keys=None,
        raw_model=create_torch_model(),
        batch_size=0,
    )


sample_request: InferenceRequest = get_request()
worker = TorchWorker()


def test_load_model(mlutils) -> None:
    fetch_model_result = FetchModelResult(sample_request.raw_model)
    load_model_result = worker.load_model(
        sample_request, fetch_model_result, mlutils.get_test_device().lower()
    )

    assert load_model_result.model(
        get_batch().to(torch_device[mlutils.get_test_device().lower()])
    ).shape == torch.Size((20, 10))


def test_transform_input(mlutils) -> None:
    fetch_input_result = FetchInputResult(
        sample_request.raw_inputs, sample_request.input_meta
    )

    transform_input_result = worker.transform_input(
        sample_request, fetch_input_result, mlutils.get_test_device().lower()
    )

    assert all(
        transformed.shape == get_batch().shape
        for transformed in transform_input_result.transformed
    )


def test_execute(mlutils) -> None:
    load_model_result = LoadModelResult(
        Net().to(torch_device[mlutils.get_test_device().lower()])
    )
    transform_result = TransformInputResult(
        [
            get_batch().to(torch_device[mlutils.get_test_device().lower()])
            for _ in range(2)
        ]
    )

    execute_result = worker.execute(sample_request, load_model_result, transform_result)

    assert all(
        result.shape == torch.Size((20, 10)) for result in execute_result.predictions
    )


def test_transform_output(mlutils):
    execute_result = ExecuteResult([torch.rand((20, 10)) for _ in range(2)])

    transformed_output = worker.transform_output(
        sample_request, execute_result, torch_device[mlutils.get_test_device().lower()]
    )

    assert transformed_output.outputs == [
        item.numpy().tobytes() for item in execute_result.predictions
    ]
    assert transformed_output.shape == None
    assert transformed_output.order == "c"
    assert transformed_output.dtype == "float32"
