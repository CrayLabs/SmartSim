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

import pathlib
import time

import pytest

dragon = pytest.importorskip("dragon")

import torch

import smartsim.error as sse
from smartsim._core.mli.infrastructure.storage.feature_store import ModelKey, TensorKey
from smartsim._core.mli.infrastructure.worker.worker import (
    InferenceRequest,
    MachineLearningWorkerCore,
    RequestBatch,
    TransformInputResult,
    TransformOutputResult,
)
from smartsim._core.utils import installed_redisai_backends

from .channel import FileSystemCommChannel
from .feature_store import FileSystemFeatureStore, MemoryFeatureStore

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon

# retrieved from pytest fixtures
is_dragon = (
    pytest.test_launcher == "dragon" if hasattr(pytest, "test_launcher") else False
)
torch_available = "torch" in installed_redisai_backends()


@pytest.fixture
def persist_torch_model(test_dir: str) -> pathlib.Path:
    ts_start = time.time_ns()
    print("Starting model file creation...")
    test_path = pathlib.Path(test_dir)
    model_path = test_path / "basic.pt"

    model = torch.nn.Linear(2, 1)
    torch.save(model, model_path)
    ts_end = time.time_ns()

    ts_elapsed = (ts_end - ts_start) / 1000000000
    print(f"Model file creation took {ts_elapsed} seconds")
    return model_path


@pytest.fixture
def persist_torch_tensor(test_dir: str) -> pathlib.Path:
    ts_start = time.time_ns()
    print("Starting model file creation...")
    test_path = pathlib.Path(test_dir)
    file_path = test_path / "tensor.pt"

    tensor = torch.randn((100, 100, 2))
    torch.save(tensor, file_path)
    ts_end = time.time_ns()

    ts_elapsed = (ts_end - ts_start) / 1000000000
    print(f"Tensor file creation took {ts_elapsed} seconds")
    return file_path


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_model_disk(persist_torch_model: pathlib.Path, test_dir: str) -> None:
    """Verify that the ML worker successfully retrieves a model
    when given a valid (file system) key"""
    worker = MachineLearningWorkerCore
    key = str(persist_torch_model)
    feature_store = FileSystemFeatureStore(test_dir)
    fsd = feature_store.descriptor
    feature_store[str(persist_torch_model)] = persist_torch_model.read_bytes()

    model_key = ModelKey(key=key, descriptor=fsd)
    request = InferenceRequest(model_key=model_key)
    batch = RequestBatch.from_requests([request], model_key)

    fetch_result = worker.fetch_model(batch, {fsd: feature_store})
    assert fetch_result.model_bytes
    assert fetch_result.model_bytes == persist_torch_model.read_bytes()


def test_fetch_model_disk_missing() -> None:
    """Verify that the ML worker fails to retrieves a model
    when given an invalid (file system) key"""
    worker = MachineLearningWorkerCore
    feature_store = MemoryFeatureStore()
    fsd = feature_store.descriptor

    key = "/path/that/doesnt/exist"

    model_key = ModelKey(key=key, descriptor=fsd)
    request = InferenceRequest(model_key=model_key)
    batch = RequestBatch.from_requests([request], model_key)

    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_model(batch, {fsd: feature_store})

    # ensure the error message includes key-identifying information
    assert key in ex.value.args[0]


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_model_feature_store(persist_torch_model: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a model
    when given a valid (file system) key"""
    worker = MachineLearningWorkerCore

    # create a key to retrieve from the feature store
    key = "test-model"

    # put model bytes into the feature store
    feature_store = MemoryFeatureStore()
    fsd = feature_store.descriptor
    feature_store[key] = persist_torch_model.read_bytes()

    model_key = ModelKey(key=key, descriptor=feature_store.descriptor)
    request = InferenceRequest(model_key=model_key)
    batch = RequestBatch.from_requests([request], model_key)

    fetch_result = worker.fetch_model(batch, {fsd: feature_store})
    assert fetch_result.model_bytes
    assert fetch_result.model_bytes == persist_torch_model.read_bytes()


def test_fetch_model_feature_store_missing() -> None:
    """Verify that the ML worker fails to retrieves a model
    when given an invalid (feature store) key"""
    worker = MachineLearningWorkerCore

    key = "some-key"
    feature_store = MemoryFeatureStore()
    fsd = feature_store.descriptor

    model_key = ModelKey(key=key, descriptor=feature_store.descriptor)
    request = InferenceRequest(model_key=model_key)
    batch = RequestBatch.from_requests([request], model_key)

    # todo: consider that raising this exception shows impl. replace...
    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_model(batch, {fsd: feature_store})

    # ensure the error message includes key-identifying information
    assert key in ex.value.args[0]


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_model_memory(persist_torch_model: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a model
    when given a valid (file system) key"""
    worker = MachineLearningWorkerCore

    key = "test-model"
    feature_store = MemoryFeatureStore()
    fsd = feature_store.descriptor
    feature_store[key] = persist_torch_model.read_bytes()

    model_key = ModelKey(key=key, descriptor=feature_store.descriptor)
    request = InferenceRequest(model_key=model_key)
    batch = RequestBatch.from_requests([request], model_key)

    fetch_result = worker.fetch_model(batch, {fsd: feature_store})
    assert fetch_result.model_bytes
    assert fetch_result.model_bytes == persist_torch_model.read_bytes()


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_input_disk(persist_torch_tensor: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (file system) key"""
    tensor_name = str(persist_torch_tensor)

    feature_store = MemoryFeatureStore()
    fsd = feature_store.descriptor
    request = InferenceRequest(input_keys=[TensorKey(key=tensor_name, descriptor=fsd)])

    model_key = ModelKey(key="test-model", descriptor=fsd)
    batch = RequestBatch.from_requests([request], model_key)

    worker = MachineLearningWorkerCore

    feature_store[tensor_name] = persist_torch_tensor.read_bytes()

    fetch_result = worker.fetch_inputs(batch, {fsd: feature_store})
    assert fetch_result.inputs[0] is not None


def test_fetch_input_disk_missing() -> None:
    """Verify that the ML worker fails to retrieves a tensor/input
    when given an invalid (file system) key"""
    worker = MachineLearningWorkerCore

    feature_store = MemoryFeatureStore()
    fsd = feature_store.descriptor
    key = "/path/that/doesnt/exist"

    request = InferenceRequest(input_keys=[TensorKey(key=key, descriptor=fsd)])

    model_key = ModelKey(key="test-model", descriptor=fsd)
    batch = RequestBatch.from_requests([request], model_key)

    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_inputs(batch, {fsd: feature_store})

    # ensure the error message includes key-identifying information
    assert key[0] in ex.value.args[0]


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_input_feature_store(persist_torch_tensor: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (feature store) key"""
    worker = MachineLearningWorkerCore

    tensor_name = "test-tensor"
    feature_store = MemoryFeatureStore()
    fsd = feature_store.descriptor

    request = InferenceRequest(input_keys=[TensorKey(key=tensor_name, descriptor=fsd)])

    # put model bytes into the feature store
    feature_store[tensor_name] = persist_torch_tensor.read_bytes()

    model_key = ModelKey(key="test-model", descriptor=fsd)
    batch = RequestBatch.from_requests([request], model_key)

    fetch_result = worker.fetch_inputs(batch, {fsd: feature_store})
    assert fetch_result.inputs[0]
    assert (
        list(fetch_result.inputs[0])[0][:10] == persist_torch_tensor.read_bytes()[:10]
    )


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_multi_input_feature_store(persist_torch_tensor: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves multiple tensor/input
    when given a valid collection of (feature store) keys"""
    worker = MachineLearningWorkerCore

    tensor_name = "test-tensor"
    feature_store = MemoryFeatureStore()
    fsd = feature_store.descriptor

    # put model bytes into the feature store
    body1 = persist_torch_tensor.read_bytes()
    feature_store[tensor_name + "1"] = body1

    body2 = b"abcdefghijklmnopqrstuvwxyz"
    feature_store[tensor_name + "2"] = body2

    body3 = b"mnopqrstuvwxyzabcdefghijkl"
    feature_store[tensor_name + "3"] = body3

    request = InferenceRequest(
        input_keys=[
            TensorKey(key=tensor_name + "1", descriptor=fsd),
            TensorKey(key=tensor_name + "2", descriptor=fsd),
            TensorKey(key=tensor_name + "3", descriptor=fsd),
        ]
    )

    model_key = ModelKey(key="test-model", descriptor=fsd)
    batch = RequestBatch.from_requests([request], model_key)

    fetch_result = worker.fetch_inputs(batch, {fsd: feature_store})

    raw_bytes = list(fetch_result.inputs[0])
    assert raw_bytes[0][:10] == persist_torch_tensor.read_bytes()[:10]
    assert raw_bytes[1][:10] == body2[:10]
    assert raw_bytes[2][:10] == body3[:10]


def test_fetch_input_feature_store_missing() -> None:
    """Verify that the ML worker fails to retrieves a tensor/input
    when given an invalid (feature store) key"""
    worker = MachineLearningWorkerCore

    key = "bad-key"
    feature_store = MemoryFeatureStore()
    fsd = feature_store.descriptor
    request = InferenceRequest(input_keys=[TensorKey(key=key, descriptor=fsd)])

    model_key = ModelKey(key="test-model", descriptor=fsd)
    batch = RequestBatch.from_requests([request], model_key)

    with pytest.raises(sse.SmartSimError) as ex:
        worker.fetch_inputs(batch, {fsd: feature_store})

    # ensure the error message includes key-identifying information
    assert key in ex.value.args[0]


@pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
def test_fetch_input_memory(persist_torch_tensor: pathlib.Path) -> None:
    """Verify that the ML worker successfully retrieves a tensor/input
    when given a valid (file system) key"""
    worker = MachineLearningWorkerCore
    feature_store = MemoryFeatureStore()
    fsd = feature_store.descriptor

    key = "test-model"
    feature_store[key] = persist_torch_tensor.read_bytes()
    request = InferenceRequest(input_keys=[TensorKey(key=key, descriptor=fsd)])

    model_key = ModelKey(key="test-model", descriptor=fsd)
    batch = RequestBatch.from_requests([request], model_key)

    fetch_result = worker.fetch_inputs(batch, {fsd: feature_store})
    assert fetch_result.inputs[0] is not None


def test_fetch_inputs_no_input_source():
    """Verify that the ML worker fails when there are no inputs provided"""
    worker = MachineLearningWorkerCore
    request1 = InferenceRequest()
    request2 = InferenceRequest()

    batch = RequestBatch.from_requests(
        [request1, request2], ModelKey("test-model", "desc")
    )

    with pytest.raises(ValueError) as exc:
        fetch_result = worker.fetch_inputs(batch, {})
    assert str(exc.value) == "No input source"


def test_fetch_inputs_no_feature_store():
    """Verify that the ML worker fails when there is no feature store"""
    worker = MachineLearningWorkerCore
    request1 = InferenceRequest(raw_inputs=[b"abcdef"])
    request2 = InferenceRequest(raw_inputs=[b"ghijkl"])

    batch = RequestBatch.from_requests(
        [request1, request2], ModelKey("test-model", "desc")
    )

    with pytest.raises(ValueError) as exc:
        fetch_result = worker.fetch_inputs(batch, {})
    assert str(exc.value) == "No feature stores provided"


def test_place_outputs(test_dir: str) -> None:
    """Verify outputs are shared using the feature store"""
    worker = MachineLearningWorkerCore

    key_name = "test-model"
    feature_store = MemoryFeatureStore()
    fsd = feature_store.descriptor

    # create a key to retrieve from the feature store
    keys = [
        TensorKey(key=key_name + "1", descriptor=fsd),
        TensorKey(key=key_name + "2", descriptor=fsd),
        TensorKey(key=key_name + "3", descriptor=fsd),
    ]

    keys2 = [
        TensorKey(key=key_name + "4", descriptor=fsd),
        TensorKey(key=key_name + "5", descriptor=fsd),
        TensorKey(key=key_name + "6", descriptor=fsd),
    ]
    data = [b"abcdef", b"ghijkl", b"mnopqr"]
    data2 = [b"stuvwx", b"yzabcd", b"efghij"]

    callback1 = FileSystemCommChannel(pathlib.Path(test_dir) / "callback1").descriptor
    callback2 = FileSystemCommChannel(pathlib.Path(test_dir) / "callback2").descriptor

    model_id = ModelKey(key="test-model", descriptor=fsd)
    request = InferenceRequest(callback_desc=callback1, output_keys=keys)
    request2 = InferenceRequest(callback_desc=callback2, output_keys=keys2)
    transform_result = TransformOutputResult(data, [1], "c", "float32")
    transform_result2 = TransformOutputResult(data2, [1], "c", "float32")

    request_batch = RequestBatch.from_requests([request, request2], model_id)

    worker.place_output(
        request_batch.output_key_refs[callback1],
        transform_result,
        {fsd: feature_store},
    )

    worker.place_output(
        request_batch.output_key_refs[callback2],
        transform_result2,
        {fsd: feature_store},
    )

    all_keys = keys + keys2
    all_data = data + data2
    for i in range(6):
        assert feature_store[all_keys[i].key] == all_data[i]


@pytest.mark.parametrize(
    "key, descriptor",
    [
        pytest.param("", "desc", id="invalid key"),
        pytest.param("key", "", id="invalid descriptor"),
    ],
)
def test_invalid_tensorkey(key, descriptor) -> None:
    with pytest.raises(ValueError):
        fsk = TensorKey(key, descriptor)
