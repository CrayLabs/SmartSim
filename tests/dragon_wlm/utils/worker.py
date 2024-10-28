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
import typing as t

import torch

import smartsim._core.mli.infrastructure.worker.worker as mliw
import smartsim.error as sse
from smartsim.log import get_logger

logger = get_logger(__name__)


class IntegratedTorchWorker(mliw.MachineLearningWorkerBase):
    """A minimum implementation of a worker that executes a PyTorch model"""

    # @staticmethod
    # def deserialize(request: InferenceRequest) -> t.List[t.Any]:
    #     # request.input_meta
    #     # request.raw_inputs
    #     return request

    @staticmethod
    def load_model(
        request: mliw.InferenceRequest, fetch_result: mliw.FetchModelResult, device: str
    ) -> mliw.LoadModelResult:
        model_bytes = fetch_result.model_bytes or request.raw_model
        if not model_bytes:
            raise ValueError("Unable to load model without reference object")

        model: torch.nn.Module = torch.load(io.BytesIO(model_bytes))
        result = mliw.LoadModelResult(model)
        return result

    @staticmethod
    def transform_input(
        request: mliw.InferenceRequest,
        fetch_result: mliw.FetchInputResult,
        device: str,
    ) -> mliw.TransformInputResult:
        # extra metadata for assembly can be found in request.input_meta
        raw_inputs = request.raw_inputs or fetch_result.inputs

        result: t.List[torch.Tensor] = []
        # should this happen here?
        # consider - fortran to c data layout
        # is there an intermediate representation before really doing torch.load?
        if raw_inputs:
            result = [torch.load(io.BytesIO(item)) for item in raw_inputs]

        return mliw.TransformInputResult(result)

    @staticmethod
    def execute(
        request: mliw.InferenceRequest,
        load_result: mliw.LoadModelResult,
        transform_result: mliw.TransformInputResult,
    ) -> mliw.ExecuteResult:
        if not load_result.model:
            raise sse.SmartSimError("Model must be loaded to execute")

        model = load_result.model
        results = [model(tensor) for tensor in transform_result.transformed]

        execute_result = mliw.ExecuteResult(results)
        return execute_result

    @staticmethod
    def transform_output(
        request: mliw.InferenceRequest,
        execute_result: mliw.ExecuteResult,
        result_device: str,
    ) -> mliw.TransformOutputResult:
        # send the original tensors...
        execute_result.predictions = [t.detach() for t in execute_result.predictions]
        # todo: solve sending all tensor metadata that coincisdes with each prediction
        return mliw.TransformOutputResult(
            execute_result.predictions, [1], "c", "float32"
        )
