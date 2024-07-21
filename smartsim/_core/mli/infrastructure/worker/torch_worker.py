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
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from .....error import SmartSimError
from .....log import get_logger
from ...mli_schemas.tensor import tensor_capnp
from .worker import (
    ExecuteResult,
    FetchInputResult,
    FetchModelResult,
    InferenceBatch,
    LoadModelResult,
    MachineLearningWorkerBase,
    TransformInputResult,
    TransformOutputResult,
)

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
logger = get_logger(__name__)


class TorchWorker(MachineLearningWorkerBase):
    """A worker that executes a PyTorch model."""

    @staticmethod
    def load_model(
        batch: InferenceBatch, fetch_result: FetchModelResult, device: str
    ) -> LoadModelResult:
        request = batch.requests[0]
        if fetch_result.model_bytes:
            model_bytes = fetch_result.model_bytes
        elif request.raw_model and request.raw_model.data:
            model_bytes = request.raw_model.data
        else:
            raise ValueError("Unable to load model without reference object")

        device_to_torch = {"cpu": "cpu", "gpu": "cuda"}
        for old, new in device_to_torch.items():
            device = device.replace(old, new)

        buffer = io.BytesIO(initial_bytes=model_bytes)
        model = torch.jit.load(buffer, map_location=device)  # type: ignore
        model.eval()
        result = LoadModelResult(model)
        return result

    @staticmethod
    def transform_input(
        batch: InferenceBatch, fetch_results: list[FetchInputResult]
    ) -> TransformInputResult:
        results: list[torch.Tensor] = []
        total_samples = 0
        slices: list[slice] = []

        all_dims: list[list[int]] = []
        all_dtypes: list[str] = []
        if fetch_results[0].meta is None:
            raise ValueError("Cannot reconstruct tensor without meta information")
        # Traverse inputs to get total number of samples and compute slices
        # Assumption: first dimension is samples, all tensors in the same input
        # have same number of samples
        # thus we only look at the first tensor for each input
        for res_idx, fetch_result in enumerate(fetch_results):
            if fetch_result.meta is None or any(
                item_meta is None for item_meta in fetch_result.meta
            ):
                raise ValueError("Cannot reconstruct tensor without meta information")
            first_tensor_desc: tensor_capnp.TensorDescriptor = fetch_result.meta[0]
            num_samples = first_tensor_desc.dimensions[0]
            slices.append(slice(total_samples, total_samples + num_samples))
            total_samples = total_samples + num_samples

            if res_idx == len(fetch_results)-1:
                # For each tensor in the last input, get remaining dimensions
                # Assumptions: all inputs have the same number of tensors and
                # last N-1 dimensions match across inputs for corresponding tensors
                # thus: resulting array will be of size (num_samples, all_other_dims)
                for item_meta in fetch_result.meta:
                    tensor_desc: tensor_capnp.TensorDescriptor = item_meta
                    tensor_dims = list(tensor_desc.dimensions)
                    all_dims.append([total_samples, *tensor_dims[1:]])
                    all_dtypes.append(str(tensor_desc.dataType))

        for result_tensor_idx, (dims, dtype) in enumerate(zip(all_dims, all_dtypes)):
            # List comprehension concatenation can be faster sometimes
            all_bytes = b"".join(
                [
                    fetch_result.inputs[result_tensor_idx]
                    for fetch_result in fetch_results
                ]
            )

            results.append(
                torch.from_numpy(
                    np.frombuffer(
                        all_bytes,
                        dtype=dtype,
                    ).reshape(dims)
                )
            )

        return TransformInputResult(results, slices)

    # @staticmethod
    # def _transform_input(
    #     batch: InferenceBatch, fetch_results: list[FetchInputResult]
    # ) -> TransformInputResult:
    #     results: list[list[torch.Tensor]] = []
    #     start = 0
    #     slices: list[slice] = []

    #     for fetch_result in fetch_results:
    #         partial_result = []
    #         if fetch_result.meta is None:
    #             raise ValueError("Cannot reconstruct tensor without meta information")
    #         for idx, (item, item_meta) in enumerate(
    #             zip(fetch_result.inputs, fetch_result.meta)
    #         ):
    #             tensor_desc: tensor_capnp.TensorDescriptor = item_meta
    #             partial_result.append(
    #                 torch.tensor(
    #                     np.frombuffer(item, dtype=str(tensor_desc.dataType))
    #                 ).reshape(tuple(dim for dim in tensor_desc.dimensions))
    #             )
    #             if idx == 0:
    #                 num_samples = tensor_desc.dimensions[0]
    #                 slices.append(slice(start, start + num_samples))
    #                 start = start + num_samples
    #         results.append(partial_result)

    #     result: list[torch.Tensor] = []
    #     if len(batch.requests) > 1:
    #         for t_idx in range(len(results[0])):
    #             result.append(
    #                 torch.concatenate(
    #                     [partial_result[t_idx] for partial_result in results]
    #                 )
    #             )
    #     else:
    #         result = results[0]

    #     return TransformInputResult(result, slices)
    # return data # note: this fails copy test!

    # pylint: disable-next=unused-argument
    @staticmethod
    def execute(
        batch: InferenceBatch,
        load_result: LoadModelResult,
        transform_result: TransformInputResult,
        device: str,
    ) -> ExecuteResult:
        if not load_result.model:
            raise SmartSimError("Model must be loaded to execute")
        device_to_torch = {"cpu": "cpu", "gpu": "cuda"}
        for old, new in device_to_torch.items():
            device = device.replace(old, new)
        model: torch.nn.Module = load_result.model
        model.eval()
        # print([tensor.shape for tensor in transform_result.transformed])
        # torch.cuda.empty_cache()
        results = [
            model(tensor.to(device)).detach() for tensor in transform_result.transformed
        ]

        transform_result.transformed = []

        execute_result = ExecuteResult(results, transform_result.slices)
        return execute_result

    @staticmethod
    def transform_output(
        batch: InferenceBatch,
        execute_result: ExecuteResult,
    ) -> list[TransformOutputResult]:
        transformed_list: list[TransformOutputResult] = []
        for result_slice in execute_result.slices:
            transformed = [
                item[result_slice].cpu().numpy().tobytes()
                for item in execute_result.predictions
            ]
            # todo: need the shape from latest schemas added here.
            transformed_list.append(
                TransformOutputResult(transformed, None, "c", "float32")
            )  # fixme

        execute_result.predictions = []

        return transformed_list
