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

import multiprocessing as mp

import pytest

dragon = pytest.importorskip("dragon")

import dragon.utils as du
from dragon.channels import Channel
from dragon.data.ddict.ddict import DDict
from dragon.fli import DragonFLIError, FLInterface

from smartsim._core.mli.infrastructure.control.workermanager import (
    WorkerManager,
    exception_handler,
)
from smartsim._core.mli.infrastructure.environmentloader import EnvironmentConfigLoader
from smartsim._core.mli.infrastructure.worker.worker import InferenceReply
from smartsim._core.mli.message_handler import MessageHandler

from .channel import FileSystemCommChannel
from .featurestore import FileSystemFeatureStore
from .worker import IntegratedTorchWorker

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


@pytest.fixture
def setup_worker_manager(test_dir, monkeypatch):
    integrated_worker = IntegratedTorchWorker()

    chan = Channel.make_process_local()
    queue = FLInterface(main_ch=chan)
    monkeypatch.setenv("SSQueue", du.B64.bytes_to_str(queue.serialize()))

    worker_manager = WorkerManager(
        EnvironmentConfigLoader(),
        integrated_worker,
        as_service=False,
        cooldown=3,
        comm_channel_type=FileSystemCommChannel,
    )

    tensor_key = MessageHandler.build_tensor_key("key")
    model = MessageHandler.build_model(b"model", "model name", "v 0.0.1")
    request = MessageHandler.build_request(
        b"channel", model, [tensor_key], [tensor_key], [], None
    )
    ser_request = MessageHandler.serialize_request(request)
    new_sender = worker_manager._task_queue.sendh(use_main_as_stream_channel=True)
    new_sender.send_bytes(ser_request)

    return worker_manager, integrated_worker


def test_execute_errors_handled(setup_worker_manager, monkeypatch: pytest.MonkeyPatch):
    """Ensures that the worker manager does not crash after a failure in the
    execute pipeline stage"""
    worker_manager, integrated_worker = setup_worker_manager

    def mock_execute():
        raise ValueError("Simulated error in execute")

    monkeypatch.setattr(integrated_worker, "execute", mock_execute)

    worker_manager._on_iteration()


def test_fetch_model_errors_handled(
    setup_worker_manager, monkeypatch: pytest.MonkeyPatch
):
    """Ensures that the worker manager does not crash after a failure in the
    fetch model pipeline stage"""
    worker_manager, integrated_worker = setup_worker_manager

    def mock_fetch_model(a, b):
        raise ValueError("Simulated error in fetch_model")

    monkeypatch.setattr(integrated_worker, "fetch_model", mock_fetch_model)

    worker_manager._on_iteration()


def test_load_model_errors_handled(
    setup_worker_manager, monkeypatch: pytest.MonkeyPatch
):
    """Ensures that the worker manager does not crash after a failure in the
    load model pipeline stage"""
    worker_manager, integrated_worker = setup_worker_manager

    def mock_load_model(a, b):
        raise ValueError("Simulated error in load_model")

    monkeypatch.setattr(integrated_worker, "load_model", mock_load_model)
    worker_manager._on_iteration()


def test_fetch_inputs_errors_handled(
    setup_worker_manager, monkeypatch: pytest.MonkeyPatch
):
    """Ensures that the worker manager does not crash after a failure in the
    fetch inputs pipeline stage"""
    worker_manager, integrated_worker = setup_worker_manager

    def mock_fetch_inputs(a, b):
        raise ValueError("Simulated error in fetch_inputs")

    monkeypatch.setattr(integrated_worker, "fetch_inputs", mock_fetch_inputs)
    worker_manager._on_iteration()


def test_transform_input_errors_handled(
    setup_worker_manager, monkeypatch: pytest.MonkeyPatch
):
    """Ensures that the worker manager does not crash after a failure in the
    transform input pipeline stage"""
    worker_manager, integrated_worker = setup_worker_manager

    def mock_transform_input(a, b):
        raise ValueError("Simulated error in transform_input")

    monkeypatch.setattr(integrated_worker, "transform_input", mock_transform_input)
    worker_manager._on_iteration()


def test_transform_output_errors_handled(
    setup_worker_manager, monkeypatch: pytest.MonkeyPatch
):
    """Ensures that the worker manager does not crash after a failure in the
    transform output pipeline stage"""
    worker_manager, integrated_worker = setup_worker_manager

    def mock_transform_output(a, b):
        raise ValueError("Simulated error in transform_output")

    monkeypatch.setattr(integrated_worker, "transform_output", mock_transform_output)
    worker_manager._on_iteration()


def test_place_output_errors_handled(
    setup_worker_manager, monkeypatch: pytest.MonkeyPatch
):
    """Ensures that the worker manager does not crash after a failure in the
    place output pipeline stage"""
    worker_manager, integrated_worker = setup_worker_manager

    def mock_place_output(a, b, c):
        raise ValueError("Simulated error in place_output")

    monkeypatch.setattr(integrated_worker, "place_output", mock_place_output)
    worker_manager._on_iteration()


def test_exception_handling_helper():
    """Ensures that the worker manager does not crash after a failure in the
    execute pipeline stage"""
    reply = InferenceReply()

    test_exception = ValueError("Test ValueError")
    exception_handler(test_exception, None, "fetching the model", reply)

    assert reply.status_enum == "fail"
    assert reply.message == "Failed while fetching the model."
