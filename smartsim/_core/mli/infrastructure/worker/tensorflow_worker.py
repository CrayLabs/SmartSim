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

import logging
import numpy as np
import numpy.typing as npt
import os
import tensorflow as tf

# pylint: disable=import-error
from dragon.managed_memory import MemoryAlloc, MemoryPool
from tensorflow.python.framework.convert_to_constants import (
    convert_var_to_const_function_in_v1,
)
from tensorflow.python.framework.ops import disable_eager_execution

tf.get_logger().setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

logger = get_logger(__name__)

disable_eager_execution()


class TensorFlowWorker(MachineLearningWorkerBase):
    """A worker that executes a TensorFlow model."""

    @staticmethod
    def load_model(
        batch: RequestBatch, fetch_result: FetchModelResult, device: str
    ) -> LoadModelResult:
        """Given a loaded MachineLearningModel, ensure it is loaded into
        device memory.

        :param request: The request that triggered the pipeline
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

        try:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(model_bytes)

            with tf.Graph().as_default() as graph, tf.device(device):
                tf.import_graph_def(graph_def, name="")

            ops = graph.get_operations()
            input_layers = []
            for op in ops:
                if op.type == "Placeholder":
                    logger.debug(
                        "Input op name: {}, output shape : {}".format(
                            op.name, op.outputs[0].get_shape()
                        )
                    )
                    input_layers.append(f"{op.name}:0")

            output_tensors = set()
            input_tensors = set()
            for op in ops:
                for x in op.inputs:
                    if x.name not in input_tensors:
                        input_tensors.add(x.name)
            for op in ops:
                if len(op.outputs) > 0:
                    x = op.outputs[0]
                    if x.name not in input_tensors:
                        logger.debug(
                            "Output tensor name: {}, tensor shape : {}, parent op type: {}".format(
                                x.name, x.get_shape(), op.type
                            )
                        )
                        output_tensors.add(x.name)

        except Exception as e:
            raise RuntimeError(
                "Failed to load and evaluate the model: "
                f"Model key {batch.model_id.key}, Device {device}"
            ) from e
        with tf.device(device):
            result = LoadModelResult(tf.compat.v1.Session(graph=graph), input_layers, list(output_tensors))
        return result

    @staticmethod
    def transform_input(
        batch: RequestBatch,
        fetch_results: list[FetchInputResult],
        mem_pool: MemoryPool,
    ) -> TransformInputResult:
        """Given a collection of data, perform a transformation on the data and put
        the raw tensor data on a MemoryPool allocation.

        :param request: The request that triggered the pipeline
        :param fetch_result: Raw outputs from fetching inputs out of a feature store
        :param mem_pool: The memory pool used to access batched input tensors
        :returns: The transformed inputs wrapped in a TransformInputResult
        :raises ValueError: If tensors cannot be reconstructed
        :raises IndexError: If index out of range
        """
        results: list[memoryview] = []
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
            try:
                joined = b"".join(
                    [
                        fetch_result.inputs[result_tensor_idx]
                        for fetch_result in fetch_results
                    ]
                )
                mem_view[:alloc_size] = joined
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
        device_to_tf = {"cpu": "/CPU", "gpu": "/GPU"}
        for old, new in device_to_tf.items():
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
                    np.frombuffer(
                        mem_alloc.get_memview()[0 : np.prod(dims) * itemsize],
                        dtype=dtype,
                    ).reshape(dims)
                )
            except IndexError as e:
                raise IndexError("Error during memory slicing") from e
            except Exception as e:
                raise ValueError("Error during tensor creation") from e

        sess = load_result.model
        try:
            with tf.device(device):
                results = sess.run(
                    load_result.outputs,
                    feed_dict={
                        input_layer: tensor
                        for input_layer, tensor in zip(load_result.inputs, tensors)
                    },
                )
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
        cpu_predictions = execute_result.predictions

        for result_slice in execute_result.slices:
            transformed = []
            for cpu_item in cpu_predictions:
                try:
                    transformed.append(cpu_item[result_slice].tobytes())

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
