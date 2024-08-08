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

import numpy as np
import torch

from .....error import SmartSimError
from .....log import get_logger
from ...mli_schemas.tensor import tensor_capnp
from .worker import (
    ExecuteResult,
    FetchInputResult,
    FetchModelResult,
    InferenceRequest,
    LoadModelResult,
    MachineLearningWorkerBase,
    TransformInputResult,
    TransformOutputResult,
)

logger = get_logger(__name__)


class TorchWorker(MachineLearningWorkerBase):
    """A worker that executes a PyTorch model."""

    @staticmethod
    def load_model(
        request: InferenceRequest, fetch_result: FetchModelResult, device: str
    ) -> LoadModelResult:
        if fetch_result.model_bytes:
            model_bytes = fetch_result.model_bytes
        elif request.raw_model and request.raw_model.data:
            model_bytes = request.raw_model.data
        else:
            raise ValueError("Unable to load model without reference object")

        device_to_torch = {"cpu": "cpu", "gpu": "cuda"}
        device = device_to_torch[device]
        buffer = io.BytesIO(initial_bytes=model_bytes)
        model = torch.jit.load(buffer, map_location=device)  # type: ignore
        result = LoadModelResult(model)
        return result

    @staticmethod
    def transform_input(
        request: InferenceRequest, fetch_result: FetchInputResult, device: str
    ) -> TransformInputResult:
        result = []

        device_to_torch = {"cpu": "cpu", "gpu": "cuda"}
        device = device_to_torch[device]
        if fetch_result.meta is None:
            raise ValueError("Cannot reconstruct tensor without meta information")
        for item, item_meta in zip(fetch_result.inputs, fetch_result.meta):
            tensor_desc: tensor_capnp.TensorDescriptor = item_meta
            result.append(
                torch.from_numpy(np.frombuffer(item, dtype=str(tensor_desc.dataType)))
                .to(device)
                .reshape(tuple(dim for dim in tensor_desc.dimensions))
            )
        return TransformInputResult(result)
        # return data # note: this fails copy test!

    @staticmethod
    def execute(
        request: InferenceRequest,
        load_result: LoadModelResult,
        transform_result: TransformInputResult,
    ) -> ExecuteResult:
        if not load_result.model:
            raise SmartSimError("Model must be loaded to execute")

        model: torch.nn.Module = load_result.model
        model.eval()
        results = [model(tensor).detach() for tensor in transform_result.transformed]

        execute_result = ExecuteResult(results)
        return execute_result

    @staticmethod
    def transform_output(
        request: InferenceRequest,
        execute_result: ExecuteResult,
        result_device: str,
    ) -> TransformOutputResult:
        if result_device != "cpu":
            transformed = [
                item.to("cpu").numpy().tobytes() for item in execute_result.predictions
            ]

            # todo: need the shape from latest schemas added here.
            return TransformOutputResult(transformed, None, "c", "float32")  # fixme

        return TransformOutputResult(
            [item.numpy().tobytes() for item in execute_result.predictions],
            None,
            "c",
            "float32",
        )  # fixme
