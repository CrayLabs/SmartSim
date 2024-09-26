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

import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_var_to_const_function_in_v1

dragon = pytest.importorskip("dragon")
import dragon.globalservices.pool as dragon_gs_pool
from dragon.managed_memory import MemoryAlloc, MemoryPool


from smartsim._core.mli.infrastructure.storage.feature_store import FeatureStoreKey
from smartsim._core.mli.infrastructure.worker.tensorflow_worker import TensorFlowWorker
from smartsim._core.mli.infrastructure.worker.worker import (
    ExecuteResult,
    FetchInputResult,
    FetchModelResult,
    InferenceRequest,
    LoadModelResult,
    RequestBatch,
    TransformInputResult,
)
from smartsim._core.mli.message_handler import MessageHandler
from smartsim.log import get_logger

logger = get_logger(__name__)
# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon



def get_batch() -> np.typing.ArrayLike:
    return np.random.randn(20, 28, 28).astype(np.float32)

def create_tf_model():
    model = keras.Sequential(
        layers=[
            keras.layers.InputLayer(input_shape=(28, 28), name="input"),
            keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
            keras.layers.Dense(128, activation="relu", name="dense"),
            keras.layers.Dense(10, activation="softmax", name="output"),
        ],
        name="FCN",
    )

    # Compile model with optimizer
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )


    real_model = tf.function(model).get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )
    with tf.compat.v1.Session() as sess:
        ffunc = convert_var_to_const_function_in_v1(real_model)
    graph_def_orig = ffunc.graph.as_graph_def()

    graph_def_str = graph_def_orig.SerializeToString()

    names = lambda l: [x.name for x in l]

    return graph_def_str, names(ffunc.inputs), names(ffunc.outputs)

tensorflow_device = {"cpu": "/CPU", "gpu": "/GPU"}


def get_request() -> InferenceRequest:

    tensors = [get_batch()]
    serialized_tensors_descriptors = [
        MessageHandler.build_tensor_descriptor("c", "float32", list(tensor.shape))
        for tensor in tensors
    ]

    return InferenceRequest(
        model_key=FeatureStoreKey(key="model", descriptor="xyz"),
        callback=None,
        raw_inputs=tensors,
        input_keys=None,
        input_meta=serialized_tensors_descriptors,
        output_keys=None,
        raw_model=create_tf_model()[0],
        batch_size=0,
    )


def get_request_batch_from_request(
    request: InferenceRequest, inputs: t.Optional[TransformInputResult] = None
) -> RequestBatch:

    return RequestBatch([request], inputs, request.model_key)


sample_request: InferenceRequest = get_request()
sample_request_batch: RequestBatch = get_request_batch_from_request(sample_request)
worker = TensorFlowWorker()


def test_load_model(mlutils) -> None:
    fetch_model_result = FetchModelResult(sample_request.raw_model)
    load_model_result = worker.load_model(
        sample_request_batch, fetch_model_result, mlutils.get_test_device().lower()
    )

    with tf.device(tensorflow_device[mlutils.get_test_device().lower()]):
        results = load_model_result.model.run(
            load_model_result.outputs,
            feed_dict=dict(zip(load_model_result.inputs, [get_batch()])),
        )

    assert results[0].shape == (20,10)


def test_transform_input(mlutils) -> None:
    fetch_input_result = FetchInputResult(
        sample_request.raw_inputs, sample_request.input_meta
    )

    mem_pool = MemoryPool.attach(dragon_gs_pool.create(1024**2).sdesc)

    transform_input_result = worker.transform_input(
        sample_request_batch, [fetch_input_result], mem_pool
    )

    batch = get_batch()
    assert transform_input_result.slices[0] == slice(0, batch.shape[0])

    tensor_index = 0
    assert tuple(transform_input_result.dims[tensor_index]) == batch.shape
    assert transform_input_result.dtypes[tensor_index] == str(batch.dtype)
    mem_alloc = MemoryAlloc.attach(transform_input_result.transformed[tensor_index])
    itemsize = batch.itemsize
    tensor = np.frombuffer(
        mem_alloc.get_memview()[
            0 : np.prod(transform_input_result.dims[tensor_index]) * itemsize
        ],
        dtype=transform_input_result.dtypes[tensor_index],
    ).reshape(transform_input_result.dims[tensor_index])

    np.testing.assert_allclose(tensor, sample_request.raw_inputs[tensor_index])

    mem_pool.destroy()


def test_execute(mlutils) -> None:

    graph_def_str, inputs, outputs = create_tf_model()
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(graph_def_str)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        load_model_result = LoadModelResult(
            tf.compat.v1.Session(graph=graph), inputs=inputs, outputs=outputs
        )

    fetch_input_result = FetchInputResult(
        sample_request.raw_inputs, sample_request.input_meta
    )

    request_batch = get_request_batch_from_request(sample_request, fetch_input_result)

    mem_pool = MemoryPool.attach(dragon_gs_pool.create(1024**2).sdesc)

    transform_result = worker.transform_input(
        request_batch, [fetch_input_result], mem_pool
    )

    execute_result = worker.execute(
        request_batch,
        load_model_result,
        transform_result,
        mlutils.get_test_device().lower(),
    )

    assert all(
        result.shape == (20, 10) for result in execute_result.predictions
    )

    mem_pool.destroy()


def test_transform_output(mlutils):
    tensors = [np.zeros((20, 10))]
    execute_result = ExecuteResult(tensors, [slice(0, 20)])

    transformed_output = worker.transform_output(sample_request_batch, execute_result)

    assert transformed_output[0].outputs == [item.tobytes() for item in tensors]
    assert transformed_output[0].shape == None
    assert transformed_output[0].order == "c"
    assert transformed_output[0].dtype == "float32"
