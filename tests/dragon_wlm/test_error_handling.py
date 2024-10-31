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
import typing as t
from unittest.mock import MagicMock

import pytest

from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel

dragon = pytest.importorskip("dragon")

import multiprocessing as mp

from dragon.channels import Channel
from dragon.data.ddict.ddict import DDict
from dragon.fli import FLInterface
from dragon.mpbridge.queues import DragonQueue

from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
from smartsim._core.mli.infrastructure.control.request_dispatcher import (
    RequestDispatcher,
)
from smartsim._core.mli.infrastructure.control.worker_manager import (
    WorkerManager,
    exception_handler,
)
from smartsim._core.mli.infrastructure.environment_loader import EnvironmentConfigLoader
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.dragon_feature_store import (
    DragonFeatureStore,
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
from smartsim._core.mli.message_handler import MessageHandler
from smartsim._core.mli.mli_schemas.response.response_capnp import ResponseBuilder

from .utils.channel import FileSystemCommChannel
from .utils.worker import IntegratedTorchWorker

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


@pytest.fixture(scope="module")
def app_feature_store(the_storage) -> FeatureStore:
    # create a standalone feature store to mimic a user application putting
    # data into an application-owned resource (app should not access backbone)
    app_fs = DragonFeatureStore(the_storage)
    return app_fs


@pytest.fixture
def setup_worker_manager_model_bytes(
    test_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    backbone_descriptor: str,
    app_feature_store: FeatureStore,
    the_worker_channel: DragonFLIChannel,
):
    integrated_worker_type = IntegratedTorchWorker

    monkeypatch.setenv(
        BackboneFeatureStore.MLI_WORKER_QUEUE, the_worker_channel.descriptor
    )
    # Put backbone descriptor into env var for the `EnvironmentConfigLoader`
    monkeypatch.setenv(BackboneFeatureStore.MLI_BACKBONE, backbone_descriptor)

    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=FileSystemCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_descriptor,
    )

    dispatcher_task_queue: mp.Queue[RequestBatch] = mp.Queue(maxsize=0)

    worker_manager = WorkerManager(
        config_loader=config_loader,
        worker_type=integrated_worker_type,
        dispatcher_queue=dispatcher_task_queue,
        as_service=False,
        cooldown=3,
    )

    tensor_key = MessageHandler.build_tensor_key("key", app_feature_store.descriptor)
    output_key = MessageHandler.build_tensor_key("key", app_feature_store.descriptor)

    callback_descriptor = FileSystemCommChannel(
        pathlib.Path(test_dir) / "callback1"
    ).descriptor

    inf_request = InferenceRequest(
        model_key=None,
        callback_desc=callback_descriptor,
        raw_inputs=None,
        input_keys=[tensor_key],
        input_meta=None,
        output_keys=[output_key],
        raw_model=b"model",
        batch_size=0,
    )

    model_id = ModelKey(key="key", descriptor=app_feature_store.descriptor)

    request_batch = RequestBatch.from_requests(
        [inf_request],
        model_id,
    )

    request_batch.inputs = TransformInputResult(
        b"transformed", [slice(0, 1)], [[1, 2]], ["float32"]
    )

    dispatcher_task_queue.put(request_batch)
    return worker_manager, integrated_worker_type


@pytest.fixture
def setup_worker_manager_model_key(
    test_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    backbone_descriptor: str,
    app_feature_store: FeatureStore,
    the_worker_channel: DragonFLIChannel,
):
    integrated_worker_type = IntegratedTorchWorker

    monkeypatch.setenv(
        BackboneFeatureStore.MLI_WORKER_QUEUE, the_worker_channel.descriptor
    )
    # Put backbone descriptor into env var for the `EnvironmentConfigLoader`
    monkeypatch.setenv(BackboneFeatureStore.MLI_BACKBONE, backbone_descriptor)

    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=FileSystemCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_descriptor,
    )

    dispatcher_task_queue: mp.Queue[RequestBatch] = mp.Queue(maxsize=0)

    worker_manager = WorkerManager(
        config_loader=config_loader,
        worker_type=integrated_worker_type,
        dispatcher_queue=dispatcher_task_queue,
        as_service=False,
        cooldown=3,
    )

    tensor_key = TensorKey(key="key", descriptor=app_feature_store.descriptor)
    output_key = TensorKey(key="key", descriptor=app_feature_store.descriptor)
    model_id = ModelKey(key="model key", descriptor=app_feature_store.descriptor)

    callback_descriptor = FileSystemCommChannel(
        pathlib.Path(test_dir) / "callback2"
    ).descriptor

    request = InferenceRequest(
        model_key=model_id,
        callback_desc=callback_descriptor,
        raw_inputs=None,
        input_keys=[tensor_key],
        input_meta=None,
        output_keys=[output_key],
        raw_model=b"model",
        batch_size=0,
    )
    request_batch = RequestBatch.from_requests(
        [request],
        model_id,
    )

    request_batch.inputs = TransformInputResult(
        b"transformed", [slice(0, 1)], [[1, 2]], ["float32"]
    )

    dispatcher_task_queue.put(request_batch)
    return worker_manager, integrated_worker_type


@pytest.fixture
def setup_request_dispatcher_model_bytes(
    test_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    backbone_descriptor: str,
    app_feature_store: FeatureStore,
    the_worker_channel: DragonFLIChannel,
):
    integrated_worker_type = IntegratedTorchWorker

    monkeypatch.setenv(
        BackboneFeatureStore.MLI_WORKER_QUEUE, the_worker_channel.descriptor
    )
    # Put backbone descriptor into env var for the `EnvironmentConfigLoader`
    monkeypatch.setenv(BackboneFeatureStore.MLI_BACKBONE, backbone_descriptor)

    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=FileSystemCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_descriptor,
    )

    request_dispatcher = RequestDispatcher(
        batch_timeout=0,
        batch_size=0,
        config_loader=config_loader,
        worker_type=integrated_worker_type,
    )
    request_dispatcher._on_start()

    tensor_key = MessageHandler.build_tensor_key("key", app_feature_store.descriptor)
    output_key = MessageHandler.build_tensor_key("key", app_feature_store.descriptor)
    model = MessageHandler.build_model(b"model", "model name", "v 0.0.1")
    request = MessageHandler.build_request(
        test_dir, model, [tensor_key], [output_key], [], None
    )
    ser_request = MessageHandler.serialize_request(request)

    request_dispatcher._incoming_channel.send(ser_request)

    return request_dispatcher, integrated_worker_type


@pytest.fixture
def setup_request_dispatcher_model_key(
    test_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    backbone_descriptor: str,
    app_feature_store: FeatureStore,
    the_worker_channel: DragonFLIChannel,
):
    integrated_worker_type = IntegratedTorchWorker

    monkeypatch.setenv(
        BackboneFeatureStore.MLI_WORKER_QUEUE, the_worker_channel.descriptor
    )
    # Put backbone descriptor into env var for the `EnvironmentConfigLoader`
    monkeypatch.setenv(BackboneFeatureStore.MLI_BACKBONE, backbone_descriptor)

    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=FileSystemCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_descriptor,
    )

    request_dispatcher = RequestDispatcher(
        batch_timeout=0,
        batch_size=0,
        config_loader=config_loader,
        worker_type=integrated_worker_type,
    )
    request_dispatcher._on_start()

    tensor_key = MessageHandler.build_tensor_key("key", app_feature_store.descriptor)
    output_key = MessageHandler.build_tensor_key("key", app_feature_store.descriptor)
    model_key = MessageHandler.build_model_key(
        key="model key", descriptor=app_feature_store.descriptor
    )
    request = MessageHandler.build_request(
        test_dir, model_key, [tensor_key], [output_key], [], None
    )
    ser_request = MessageHandler.serialize_request(request)

    request_dispatcher._incoming_channel.send(ser_request)

    return request_dispatcher, integrated_worker_type


def mock_pipeline_stage(
    monkeypatch: pytest.MonkeyPatch,
    integrated_worker: MachineLearningWorkerBase,
    stage: str,
) -> t.Callable[[t.Any], ResponseBuilder]:
    def mock_stage(*args: t.Any, **kwargs: t.Any) -> None:
        raise ValueError(f"Simulated error in {stage}")

    monkeypatch.setattr(integrated_worker, stage, mock_stage)
    mock_reply_fn = MagicMock()
    mock_response = MagicMock()
    mock_response.schema.node.displayName = "Response"
    mock_reply_fn.return_value = mock_response

    monkeypatch.setattr(
        "smartsim._core.mli.infrastructure.control.error_handling.build_failure_reply",
        mock_reply_fn,
    )

    mock_reply_channel = MagicMock()
    mock_reply_channel.send = MagicMock()

    def mock_exception_handler(
        exc: Exception, reply_channel: CommChannelBase, failure_message: str
    ) -> None:
        exception_handler(exc, mock_reply_channel, failure_message)

    monkeypatch.setattr(
        "smartsim._core.mli.infrastructure.control.worker_manager.exception_handler",
        mock_exception_handler,
    )

    monkeypatch.setattr(
        "smartsim._core.mli.infrastructure.control.request_dispatcher.exception_handler",
        mock_exception_handler,
    )

    return mock_reply_fn


@pytest.mark.parametrize(
    "setup_worker_manager",
    [
        pytest.param("setup_worker_manager_model_bytes"),
        pytest.param("setup_worker_manager_model_key"),
    ],
)
@pytest.mark.parametrize(
    "stage, error_message",
    [
        pytest.param(
            "fetch_model",
            "Error loading model on device or getting device.",
            id="fetch model",
        ),
        pytest.param(
            "load_model",
            "Error loading model on device or getting device.",
            id="load model",
        ),
        pytest.param("execute", "Error while executing.", id="execute"),
        pytest.param(
            "transform_output",
            "Error while transforming the output.",
            id="transform output",
        ),
        pytest.param(
            "place_output", "Error while placing the output.", id="place output"
        ),
    ],
)
def test_wm_pipeline_stage_errors_handled(
    request: pytest.FixtureRequest,
    setup_worker_manager: str,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error_message: str,
) -> None:
    """Ensures that the worker manager does not crash after a failure in various pipeline stages"""
    worker_manager, integrated_worker_type = request.getfixturevalue(
        setup_worker_manager
    )
    integrated_worker = worker_manager._worker

    worker_manager._on_start()
    device = worker_manager._device_manager._device
    mock_reply_fn = mock_pipeline_stage(monkeypatch, integrated_worker, stage)

    if stage not in ["fetch_model"]:
        monkeypatch.setattr(
            integrated_worker,
            "fetch_model",
            MagicMock(return_value=FetchModelResult(b"result_bytes")),
        )
    if stage not in ["fetch_model", "load_model"]:
        monkeypatch.setattr(
            integrated_worker,
            "load_model",
            MagicMock(return_value=LoadModelResult(b"result_bytes")),
        )
        monkeypatch.setattr(
            device,
            "get_model",
            MagicMock(return_value=b"result_bytes"),
        )
    if stage not in [
        "fetch_model",
        "execute",
    ]:
        monkeypatch.setattr(
            integrated_worker,
            "execute",
            MagicMock(return_value=ExecuteResult(b"result_bytes", [slice(0, 1)])),
        )
    if stage not in [
        "fetch_model",
        "execute",
        "transform_output",
    ]:
        monkeypatch.setattr(
            integrated_worker,
            "transform_output",
            MagicMock(
                return_value=[TransformOutputResult(b"result", [], "c", "float32")]
            ),
        )

    worker_manager._on_iteration()

    mock_reply_fn.assert_called_once()
    mock_reply_fn.assert_called_with("fail", error_message)


@pytest.mark.parametrize(
    "setup_request_dispatcher",
    [
        pytest.param("setup_request_dispatcher_model_bytes"),
        pytest.param("setup_request_dispatcher_model_key"),
    ],
)
@pytest.mark.parametrize(
    "stage, error_message",
    [
        pytest.param(
            "fetch_inputs",
            "Error fetching input.",
            id="fetch input",
        ),
        pytest.param(
            "transform_input",
            "Error transforming input.",
            id="transform input",
        ),
    ],
)
def test_dispatcher_pipeline_stage_errors_handled(
    request: pytest.FixtureRequest,
    setup_request_dispatcher: str,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error_message: str,
) -> None:
    """Ensures that the request dispatcher does not crash after a failure in various pipeline stages"""
    request_dispatcher, integrated_worker_type = request.getfixturevalue(
        setup_request_dispatcher
    )
    integrated_worker = request_dispatcher._worker

    mock_reply_fn = mock_pipeline_stage(monkeypatch, integrated_worker, stage)

    if stage not in ["fetch_inputs"]:
        monkeypatch.setattr(
            integrated_worker,
            "fetch_inputs",
            MagicMock(return_value=[FetchInputResult(result=[b"result"], meta=None)]),
        )

    request_dispatcher._on_iteration()

    mock_reply_fn.assert_called_once()
    mock_reply_fn.assert_called_with("fail", error_message)


def test_exception_handling_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensures that the worker manager does not crash after a failure in the
    execute pipeline stage"""

    mock_reply_channel = MagicMock()
    mock_reply_channel.send = MagicMock()

    mock_reply_fn = MagicMock()

    mock_response = MagicMock()
    mock_response.schema.node.displayName = "Response"
    mock_reply_fn.return_value = mock_response

    monkeypatch.setattr(
        "smartsim._core.mli.infrastructure.control.error_handling.build_failure_reply",
        mock_reply_fn,
    )

    test_exception = ValueError("Test ValueError")
    exception_handler(
        test_exception, mock_reply_channel, "Failure while fetching the model."
    )

    mock_reply_fn.assert_called_once()
    mock_reply_fn.assert_called_with("fail", "Failure while fetching the model.")


def test_dragon_feature_store_invalid_storage():
    """Verify that attempting to create a DragonFeatureStore without storage fails."""
    storage = None

    with pytest.raises(ValueError) as ex:
        DragonFeatureStore(storage)

    assert "storage" in ex.value.args[0].lower()
    assert "required" in ex.value.args[0].lower()
