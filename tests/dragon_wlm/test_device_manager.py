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

import typing as t

import pytest

dragon = pytest.importorskip("dragon")

from smartsim._core.mli.infrastructure.control.device_manager import (
    DeviceManager,
    WorkerDevice,
)
from smartsim._core.mli.infrastructure.storage.feature_store import (
    FeatureStore,
    ModelKey,
    TensorKey,
)
from smartsim._core.mli.infrastructure.worker.worker import (
    ExecuteResult,
    FetchInputResult,
    FetchModelResult,
    InferenceRequest,
    LoadModelResult,
    MachineLearningWorkerBase,
    RequestBatch,
    TransformInputResult,
    TransformOutputResult,
)

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


class MockWorker(MachineLearningWorkerBase):
    @staticmethod
    def fetch_model(
        batch: RequestBatch, feature_stores: t.Dict[str, FeatureStore]
    ) -> FetchModelResult:
        if batch.has_raw_model:
            return FetchModelResult(batch.raw_model)
        return FetchModelResult(b"fetched_model")

    @staticmethod
    def load_model(
        batch: RequestBatch, fetch_result: FetchModelResult, device: str
    ) -> LoadModelResult:
        return LoadModelResult(fetch_result.model_bytes)

    @staticmethod
    def transform_input(
        batch: RequestBatch,
        fetch_results: list[FetchInputResult],
        mem_pool: "MemoryPool",
    ) -> TransformInputResult:
        return TransformInputResult(b"result", [slice(0, 1)], [[1, 2]], ["float32"])

    @staticmethod
    def execute(
        batch: RequestBatch,
        load_result: LoadModelResult,
        transform_result: TransformInputResult,
        device: str,
    ) -> ExecuteResult:
        return ExecuteResult(b"result", [slice(0, 1)])

    @staticmethod
    def transform_output(
        batch: RequestBatch, execute_result: ExecuteResult
    ) -> t.List[TransformOutputResult]:
        return [TransformOutputResult(b"result", None, "c", "float32")]


def test_worker_device():
    worker_device = WorkerDevice("gpu:0")
    assert worker_device.name == "gpu:0"

    model_key = "my_model_key"
    model = b"the model"

    worker_device.add_model(model_key, model)

    assert model_key in worker_device
    assert worker_device.get_model(model_key) == model
    worker_device.remove_model(model_key)

    assert model_key not in worker_device


def test_device_manager_model_in_request():

    worker_device = WorkerDevice("gpu:0")
    device_manager = DeviceManager(worker_device)

    worker = MockWorker()

    tensor_key = TensorKey(key="key", descriptor="desc")
    output_key = TensorKey(key="key", descriptor="desc")
    model_key = ModelKey(key="model key", descriptor="desc")

    request = InferenceRequest(
        model_key=model_key,
        callback_desc=None,
        raw_inputs=None,
        input_keys=[tensor_key],
        input_meta=None,
        output_keys=[output_key],
        raw_model=b"raw model",
        batch_size=0,
    )

    request_batch = RequestBatch.from_requests(
        [request],
        model_key,
    )

    request_batch.inputs = TransformInputResult(
        b"transformed", [slice(0, 1)], [[1, 2]], ["float32"]
    )

    with device_manager.get_device(
        worker=worker, batch=request_batch, feature_stores={}
    ) as returned_device:

        assert returned_device == worker_device
        assert worker_device.get_model(model_key.key) == b"raw model"

    assert model_key.key not in worker_device


def test_device_manager_model_key():

    worker_device = WorkerDevice("gpu:0")
    device_manager = DeviceManager(worker_device)

    worker = MockWorker()

    tensor_key = TensorKey(key="key", descriptor="desc")
    output_key = TensorKey(key="key", descriptor="desc")
    model_key = ModelKey(key="model key", descriptor="desc")

    request = InferenceRequest(
        model_key=model_key,
        callback_desc=None,
        raw_inputs=None,
        input_keys=[tensor_key],
        input_meta=None,
        output_keys=[output_key],
        raw_model=None,
        batch_size=0,
    )

    request_batch = RequestBatch.from_requests(
        [request],
        model_key,
    )

    request_batch.inputs = TransformInputResult(
        b"transformed", [slice(0, 1)], [[1, 2]], ["float32"]
    )

    with device_manager.get_device(
        worker=worker, batch=request_batch, feature_stores={}
    ) as returned_device:

        assert returned_device == worker_device
        assert worker_device.get_model(model_key.key) == b"fetched_model"

    assert model_key.key in worker_device
