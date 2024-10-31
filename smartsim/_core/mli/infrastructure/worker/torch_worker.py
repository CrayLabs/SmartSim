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
from .worker import (
    ExecuteResult,
    FetchInputResult,
    FetchModelResult,
    LoadModelResult,
    MachineLearningWorkerBase,
    RequestBatch,
    TensorMeta,
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
        """Given a loaded MachineLearningModel, ensure it is loaded into
        device memory.

        :param batch: The batch that triggered the pipeline
        :param fetch_result: The fetched model
        :param device: The device on which the model must be placed
        :returns: LoadModelResult wrapping the model loaded for the request
        :raises ValueError: If model reference object is not found
        :raises RuntimeError: If loading and evaluating the model failed
        """
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
        try:
            with torch.no_grad():
                model = torch.jit.load(buffer, map_location=device)  # type: ignore
                model.eval()
        except Exception as e:
            raise RuntimeError(
                "Failed to load and evaluate the model: "
                f"Model key {batch.model_id.key}, Device {device}"
            ) from e
        result = LoadModelResult(model)
        return result

    @staticmethod
    def transform_input(
        batch: RequestBatch,
        fetch_results: FetchInputResult,
        mem_pool: MemoryPool,
    ) -> TransformInputResult:
        """Given a collection of data, perform a transformation on the data and put
        the raw tensor data on a MemoryPool allocation.

        :param batch: The batch that triggered the pipeline
        :param fetch_result: Raw outputs from fetching inputs out of a feature store
        :param mem_pool: The memory pool used to access batched input tensors
        :returns: The transformed inputs wrapped in a TransformInputResult
        :raises ValueError: If tensors cannot be reconstructed
        :raises IndexError: If index out of range
        """
        results: list[torch.Tensor] = []
        total_samples = 0
        slices: list[slice] = []

        all_dims: list[list[int]] = []
        all_dtypes: list[str] = []
        if fetch_results.meta is None:
            raise ValueError("Cannot reconstruct tensor without meta information")
        # Traverse inputs to get total number of samples and compute slices
        # Assumption: first dimension is samples, all tensors in the same input
        # have same number of samples
        # thus we only look at the first tensor for each input
        for res_idx, res_meta_list in enumerate(fetch_results.meta):
            if res_meta_list is None or any(
                item_meta is None for item_meta in res_meta_list
            ):
                raise ValueError("Cannot reconstruct tensor without meta information")
            first_tensor_desc: TensorMeta = res_meta_list[0]  # type: ignore
            num_samples = first_tensor_desc.dimensions[0]
            slices.append(slice(total_samples, total_samples + num_samples))
            total_samples = total_samples + num_samples

            if res_idx == len(fetch_results.meta) - 1:
                # For each tensor in the last input, get remaining dimensions
                # Assumptions: all inputs have the same number of tensors and
                # last N-1 dimensions match across inputs for corresponding tensors
                # thus: resulting array will be of size (num_samples, all_other_dims)
                for item_meta in res_meta_list:
                    tensor_desc: TensorMeta = item_meta  # type: ignore
                    tensor_dims = tensor_desc.dimensions
                    all_dims.append([total_samples, *tensor_dims[1:]])
                    all_dtypes.append(tensor_desc.datatype)

        for result_tensor_idx, (dims, dtype) in enumerate(zip(all_dims, all_dtypes)):
            itemsize = np.empty((1), dtype=dtype).itemsize
            alloc_size = int(np.prod(dims) * itemsize)
            mem_alloc = mem_pool.alloc(alloc_size)
            mem_view = mem_alloc.get_memview()
            try:
                mem_view[:alloc_size] = b"".join(
                    [
                        fetch_result[result_tensor_idx]
                        for fetch_result in fetch_results.inputs
                    ]
                )
            except IndexError as e:
                raise IndexError(
                    "Error accessing elements in fetch_result.inputs "
                    f"with index {result_tensor_idx}"
                ) from e

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
        """Execute an ML model on inputs transformed for use by the model.

        :param batch: The batch of requests that triggered the pipeline
        :param load_result: The result of loading the model onto device memory
        :param transform_result: The result of transforming inputs for model consumption
        :param device: The device on which the model will be executed
        :returns: The result of inference wrapped in an ExecuteResult
        :raises SmartSimError: If model is not loaded
        :raises IndexError: If memory slicing is out of range
        :raises ValueError: If tensor creation fails or is unable to evaluate the model
        """
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
            try:
                tensors.append(
                    torch.from_numpy(
                        np.frombuffer(
                            mem_alloc.get_memview()[0 : np.prod(dims) * itemsize],
                            dtype=dtype,
                        ).reshape(dims)
                    )
                )
            except IndexError as e:
                raise IndexError("Error during memory slicing") from e
            except Exception as e:
                raise ValueError("Error during tensor creation") from e

        model: torch.nn.Module = load_result.model
        try:
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
        except Exception as e:
            raise ValueError(
                f"Error while evaluating the model: Model {batch.model_id.key}"
            ) from e

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
        """Given inference results, perform transformations required to
        transmit results to the requestor.

        :param batch: The batch of requests that triggered the pipeline
        :param execute_result: The result of inference wrapped in an ExecuteResult
        :returns: A list of transformed outputs
        :raises IndexError: If indexing is out of range
        :raises ValueError: If transforming output fails
        """
        transformed_list: list[TransformOutputResult] = []
        cpu_predictions = [
            prediction.cpu() for prediction in execute_result.predictions
        ]
        for result_slice in execute_result.slices:
            transformed = []
            for cpu_item in cpu_predictions:
                try:
                    transformed.append(cpu_item[result_slice].numpy().tobytes())

                    # todo: need the shape from latest schemas added here.
                    transformed_list.append(
                        TransformOutputResult(transformed, None, "c", "float32")
                    )  # fixme
                except IndexError as e:
                    raise IndexError(
                        f"Error accessing elements: result_slice {result_slice}"
                    ) from e
                except Exception as e:
                    raise ValueError("Error transforming output") from e

        execute_result.predictions = []

        return transformed_list
