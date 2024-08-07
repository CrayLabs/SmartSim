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

from unittest.mock import MagicMock

import pytest

dragon = pytest.importorskip("dragon")

import dragon.utils as du
from dragon.channels import Channel
from dragon.data.ddict.ddict import DDict
from dragon.fli import FLInterface

from smartsim._core.mli.comm.channel.dragonfli import DragonFLIChannel
from smartsim._core.mli.infrastructure.control.workermanager import (
    WorkerManager,
    exception_handler,
)
from smartsim._core.mli.infrastructure.environmentloader import EnvironmentConfigLoader
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore
from smartsim._core.mli.infrastructure.worker.worker import (
    ExecuteResult,
    FetchInputResult,
    FetchModelResult,
    InferenceReply,
    LoadModelResult,
    TransformInputResult,
    TransformOutputResult,
)
from smartsim._core.mli.message_handler import MessageHandler

from .utils.channel import FileSystemCommChannel
from .utils.worker import IntegratedTorchWorker

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


@pytest.fixture
def backbone_descriptor() -> str:
    # create a shared backbone featurestore
    feature_store = DragonFeatureStore(DDict())
    return feature_store.descriptor


@pytest.fixture
def app_feature_store() -> FeatureStore:
    # create a standalone feature store to mimic a user application putting
    # data into an application-owned resource (app should not access backbone)
    app_fs = DragonFeatureStore(DDict())
    return app_fs


@pytest.fixture
def setup_worker_manager_model_bytes(
    test_dir,
    monkeypatch: pytest.MonkeyPatch,
    backbone_descriptor: str,
    app_feature_store: FeatureStore,
):
    integrated_worker = IntegratedTorchWorker()

    chan = Channel.make_process_local()
    queue = FLInterface(main_ch=chan)
    monkeypatch.setenv("SS_REQUEST_QUEUE", du.B64.bytes_to_str(queue.serialize()))
    # Put backbone descriptor into env var for the `EnvironmentConfigLoader`
    monkeypatch.setenv("SS_INFRA_BACKBONE", backbone_descriptor)

    worker_manager = WorkerManager(
        EnvironmentConfigLoader(
            featurestore_factory=DragonFeatureStore.from_descriptor,
            callback_factory=FileSystemCommChannel.from_descriptor,
            queue_factory=DragonFLIChannel.from_descriptor,
        ),
        integrated_worker,
        as_service=False,
        cooldown=3,
    )

    tensor_key = MessageHandler.build_tensor_key("key", app_feature_store.descriptor)
    output_key = MessageHandler.build_tensor_key("key", app_feature_store.descriptor)
    model = MessageHandler.build_model(b"model", "model name", "v 0.0.1")
    request = MessageHandler.build_request(
        test_dir, model, [tensor_key], [output_key], [], None
    )
    ser_request = MessageHandler.serialize_request(request)
    worker_manager._task_queue.send(ser_request)

    return worker_manager, integrated_worker


@pytest.fixture
def setup_worker_manager_model_key(
    test_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    backbone_descriptor: str,
    app_feature_store: FeatureStore,
):
    integrated_worker = IntegratedTorchWorker()

    chan = Channel.make_process_local()
    queue = FLInterface(main_ch=chan)
    monkeypatch.setenv("SS_REQUEST_QUEUE", du.B64.bytes_to_str(queue.serialize()))
    # Put backbone descriptor into env var for the `EnvironmentConfigLoader`
    monkeypatch.setenv("SS_INFRA_BACKBONE", backbone_descriptor)

    worker_manager = WorkerManager(
        EnvironmentConfigLoader(
            featurestore_factory=DragonFeatureStore.from_descriptor,
            callback_factory=FileSystemCommChannel.from_descriptor,
            queue_factory=DragonFLIChannel.from_descriptor,
        ),
        integrated_worker,
        as_service=False,
        cooldown=3,
    )

    tensor_key = MessageHandler.build_tensor_key("key", app_feature_store.descriptor)
    output_key = MessageHandler.build_tensor_key("key", app_feature_store.descriptor)
    model_key = MessageHandler.build_model_key(
        "model key", app_feature_store.descriptor
    )
    request = MessageHandler.build_request(
        test_dir, model_key, [tensor_key], [output_key], [], None
    )
    ser_request = MessageHandler.serialize_request(request)
    worker_manager._task_queue.send(ser_request)

    return worker_manager, integrated_worker


def mock_pipeline_stage(monkeypatch: pytest.MonkeyPatch, integrated_worker, stage):
    def mock_stage(*args, **kwargs):
        raise ValueError(f"Simulated error in {stage}")

    monkeypatch.setattr(integrated_worker, stage, mock_stage)
    mock_reply_fn = MagicMock()
    monkeypatch.setattr(
        "smartsim._core.mli.infrastructure.control.workermanager.build_failure_reply",
        mock_reply_fn,
    )

    def mock_exception_handler(exc, reply_channel, failure_message):
        return exception_handler(exc, None, failure_message)

    monkeypatch.setattr(
        "smartsim._core.mli.infrastructure.control.workermanager.exception_handler",
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
            "fetch_model", "Failed while fetching the model.", id="fetch model"
        ),
        pytest.param(
            "load_model",
            "Failed while loading model from feature store.",
            id="load model",
        ),
        pytest.param(
            "fetch_inputs", "Failed while fetching the inputs.", id="fetch inputs"
        ),
        pytest.param(
            "transform_input",
            "Failed while transforming the input.",
            id="transform inputs",
        ),
        pytest.param("execute", "Failed while executing.", id="execute"),
        pytest.param(
            "transform_output",
            "Failed while transforming the output.",
            id="transform output",
        ),
        pytest.param(
            "place_output", "Failed while placing the output.", id="place output"
        ),
    ],
)
def test_pipeline_stage_errors_handled(
    request,
    setup_worker_manager,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error_message: str,
):
    """Ensures that the worker manager does not crash after a failure in various pipeline stages"""
    worker_manager, integrated_worker = request.getfixturevalue(setup_worker_manager)
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
    if stage not in ["fetch_model", "load_model", "fetch_inputs"]:
        monkeypatch.setattr(
            integrated_worker,
            "fetch_inputs",
            MagicMock(return_value=FetchInputResult([b"result_bytes"], None)),
        )
    if stage not in ["fetch_model", "load_model", "fetch_inputs", "transform_input"]:
        monkeypatch.setattr(
            integrated_worker,
            "transform_input",
            MagicMock(return_value=TransformInputResult(b"result_bytes")),
        )
    if stage not in [
        "fetch_model",
        "load_model",
        "fetch_inputs",
        "transform_input",
        "execute",
    ]:
        monkeypatch.setattr(
            integrated_worker,
            "execute",
            MagicMock(return_value=ExecuteResult(b"result_bytes")),
        )
    if stage not in [
        "fetch_model",
        "load_model",
        "fetch_inputs",
        "transform_input",
        "execute",
        "transform_output",
    ]:
        monkeypatch.setattr(
            integrated_worker,
            "transform_output",
            MagicMock(
                return_value=TransformOutputResult(b"result", [], "c", "float32")
            ),
        )

    worker_manager._on_iteration()

    mock_reply_fn.assert_called_once()
    mock_reply_fn.assert_called_with("fail", error_message)


def test_exception_handling_helper(monkeypatch: pytest.MonkeyPatch):
    """Ensures that the worker manager does not crash after a failure in the
    execute pipeline stage"""
    reply = InferenceReply()

    mock_reply_fn = MagicMock()
    monkeypatch.setattr(
        "smartsim._core.mli.infrastructure.control.workermanager.build_failure_reply",
        mock_reply_fn,
    )

    test_exception = ValueError("Test ValueError")
    exception_handler(test_exception, None, "Failure while fetching the model.")

    mock_reply_fn.assert_called_once()
    mock_reply_fn.assert_called_with("fail", "Failure while fetching the model.")
