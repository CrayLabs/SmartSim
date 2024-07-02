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
import multiprocessing as mp

import pytest
import torch

from smartsim._core.mli.infrastructure.control.workermanager import WorkerManager
from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore
from smartsim._core.mli.message_handler import MessageHandler
from smartsim.log import get_logger

from .channel import FileSystemCommChannel
from .featurestore import FileSystemFeatureStore
from .worker import IntegratedTorchWorker

work_queue: "mp.Queue[bytes]" = mp.Queue()
integrated_worker = IntegratedTorchWorker()
file_system_store = FileSystemFeatureStore()


worker_manager = WorkerManager(
    work_queue,
    integrated_worker,
    file_system_store,
    as_service=False,
    cooldown=10,
    comm_channel_type=FileSystemCommChannel,
)
tensor_key = MessageHandler.build_tensor_key("key")
request = MessageHandler.build_request(
    b"channel", b"model", [tensor_key], [tensor_key], [], None
)
ser_request = MessageHandler.serialize_request(request)


def test_execute_errors_handled(monkeypatch):
    def mock_execute():
        raise ValueError("Simulated error in execute")

    work_queue.put(ser_request)

    monkeypatch.setattr(integrated_worker, "execute", mock_execute)

    worker_manager._on_iteration()


def test_fetch_model_errors_handled(monkeypatch):
    def mock_fetch_model(a, b):
        raise ValueError("Simulated error in fetch_model")

    work_queue.put(ser_request)

    monkeypatch.setattr(integrated_worker, "fetch_model", mock_fetch_model)

    worker_manager._on_iteration()


def test_load_model_errors_handled(monkeypatch):
    def mock_load_model(a, b):
        raise ValueError("Simulated error in load_model")

    work_queue.put(ser_request)

    monkeypatch.setattr(integrated_worker, "load_model", mock_load_model)
    worker_manager._on_iteration()


def test_fetch_inputs_errors_handled(monkeypatch):
    def mock_fetch_inputs(a, b):
        raise ValueError("Simulated error in fetch_inputs")

    work_queue.put(ser_request)

    monkeypatch.setattr(integrated_worker, "fetch_inputs", mock_fetch_inputs)
    worker_manager._on_iteration()


def test_transform_input_errors_handled(monkeypatch):
    def mock_transform_input(a, b):
        raise ValueError("Simulated error in transform_input")

    work_queue.put(ser_request)

    monkeypatch.setattr(integrated_worker, "transform_input", mock_transform_input)
    worker_manager._on_iteration()


def test_transform_output_errors_handled(monkeypatch):
    def mock_transform_output(a, b):
        raise ValueError("Simulated error in transform_output")

    work_queue.put(ser_request)

    monkeypatch.setattr(integrated_worker, "transform_output", mock_transform_output)
    worker_manager._on_iteration()


def test_place_output_errors_handled(monkeypatch, caplog):
    def mock_place_output(a, b, c):
        raise ValueError("Simulated error in place_output")

    work_queue.put(ser_request)

    monkeypatch.setattr(integrated_worker, "place_output", mock_place_output)
    worker_manager._on_iteration()

    with caplog.at_level(logging.ERROR):
        worker_manager._on_iteration()

    # Check if the expected error message was logged
    assert any("Simulated error in place_output" in message for message in caplog.text)
