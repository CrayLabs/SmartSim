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

# pylint: disable=import-error
from dragon.managed_memory import MemoryAlloc, MemoryPool

from .....error import SmartSimError
from .....log import get_logger
from ...mli_schemas.tensor import tensor_capnp
from .worker import (
    ExecuteResult,
    FetchInputResult,
    FetchModelResult,
    LoadModelResult,
    MachineLearningWorkerBase,
    RequestBatch,
    TransformInputResult,
    TransformOutputResult,
)

# pylint: enable=import-error


torch.set_num_threads(1)
torch.set_num_interop_threads(4)
logger = get_logger(__name__)


class TorchWorker(MachineLearningWorkerBase):
    """A worker that executes a PyTorch model."""

    @staticmethod
    def load_model(
        batch: RequestBatch, fetch_result: FetchModelResult, device: str
    ) -> LoadModelResult:
        if fetch_result.model_bytes:
            model_bytes = fetch_result.model_bytes
        elif batch.raw_model and batch.raw_model.data:
            model_bytes = batch.raw_model.data
        else:
            raise ValueError("Unable to load model without reference object")

        device_to_torch = {"cpu": "cpu", "gpu": "cuda"}
        for old, new in device_to_torch.items():
            device = device.replace(old, new)

        buffer = io.BytesIO(initial_bytes=model_bytes)
        with torch.no_grad():
            model = torch.jit.load(buffer, map_location=device)  # type: ignore
            model.eval()
        result = LoadModelResult(model)
        return result

    @staticmethod
    def transform_input(
        batch: RequestBatch,
        fetch_results: list[FetchInputResult],
        mem_pool: MemoryPool,
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

            if res_idx == len(fetch_results) - 1:
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
            itemsize = np.empty((1), dtype=dtype).itemsize
            alloc_size = int(np.prod(dims) * itemsize)
            mem_alloc = mem_pool.alloc(alloc_size)
            mem_view = mem_alloc.get_memview()
            mem_view[:alloc_size] = b"".join(
                [
                    fetch_result.inputs[result_tensor_idx]
                    for fetch_result in fetch_results
                ]
            )

            results.append(mem_alloc.serialize())

        return TransformInputResult(results, slices, all_dims, all_dtypes)

    # pylint: disable-next=unused-argument
    @staticmethod
    def execute(
        batch: RequestBatch,
        load_result: LoadModelResult,
        transform_result: TransformInputResult,
        device: str,
    ) -> ExecuteResult:
        if not load_result.model:
            raise SmartSimError("Model must be loaded to execute")
        device_to_torch = {"cpu": "cpu", "gpu": "cuda"}
        for old, new in device_to_torch.items():
            device = device.replace(old, new)

        tensors = []
        mem_allocs = []
        for transformed, dims, dtype in zip(
            transform_result.transformed, transform_result.dims, transform_result.dtypes
        ):
            mem_alloc = MemoryAlloc.attach(transformed)
            mem_allocs.append(mem_alloc)
            itemsize = np.empty((1), dtype=dtype).itemsize
            tensors.append(
                torch.from_numpy(
                    np.frombuffer(
                        mem_alloc.get_memview()[0 : np.prod(dims) * itemsize],
                        dtype=dtype,
                    ).reshape(dims)
                )
            )

        model: torch.nn.Module = load_result.model
        with torch.no_grad():
            model.eval()
            results = [
                model(
                    *[
                        tensor.to(device, non_blocking=True).detach()
                        for tensor in tensors
                    ]
                )
            ]

        transform_result.transformed = []

        execute_result = ExecuteResult(results, transform_result.slices)
        for mem_alloc in mem_allocs:
            mem_alloc.free()
        return execute_result

    @staticmethod
    def transform_output(
        batch: RequestBatch,
        execute_result: ExecuteResult,
    ) -> list[TransformOutputResult]:
        transformed_list: list[TransformOutputResult] = []
        cpu_predictions = [
            prediction.cpu() for prediction in execute_result.predictions
        ]
        for result_slice in execute_result.slices:
            transformed = []
            for cpu_item in cpu_predictions:
                transformed.append(cpu_item[result_slice].numpy().tobytes())

                # todo: need the shape from latest schemas added here.
                transformed_list.append(
                    TransformOutputResult(transformed, None, "c", "float32")
                )  # fixme

        execute_result.predictions = []

        return transformed_list
