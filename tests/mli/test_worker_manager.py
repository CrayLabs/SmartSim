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
import pathlib
import time
import typing as t

import pytest
import torch

dragon = pytest.importorskip("dragon")

from smartsim._core.mli.infrastructure.control.workermanager import (
    EnvironmentConfigLoader,
    WorkerManager,
)
from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore
from smartsim._core.mli.message_handler import MessageHandler
from smartsim.log import get_logger

from ..mli_utils.channel import FileSystemCommChannel
from ..mli_utils.featurestore import FileSystemFeatureStore
from ..mli_utils.worker import IntegratedTorchWorker

logger = get_logger(__name__)
# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


def mock_work(worker_manager_queue: "mp.Queue[bytes]") -> None:
    """Mock event producer for triggering the inference pipeline"""
    # todo: move to unit tests
    while True:
        time.sleep(1)
        # 1. for demo, ignore upstream and just put stuff into downstream
        # 2. for demo, only one downstream but we'd normally have to filter
        #       msg content and send to the correct downstream (worker) queue
        timestamp = time.time_ns()
        output_dir = "/lus/bnchlu1/mcbridch/code/ss/_tmp"
        output_path = pathlib.Path(output_dir)

        mock_channel = output_path / f"brainstorm-{timestamp}.txt"
        mock_model = output_path / "brainstorm.pt"

        output_path.mkdir(parents=True, exist_ok=True)
        mock_channel.touch()
        mock_model.touch()

        msg = f"PyTorch:{mock_model}:MockInputToReplace:{mock_channel}"
        worker_manager_queue.put(msg.encode("utf-8"))


def persist_model_file(model_path: pathlib.Path) -> pathlib.Path:
    """Create a simple torch model and persist to disk for
    testing purposes.

    TODO: remove once unit tests are in place"""
    # test_path = pathlib.Path(work_dir)
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)

    model_path.unlink(missing_ok=True)
    # model_path = test_path / "basic.pt"

    model = torch.nn.Linear(2, 1)
    torch.save(model, model_path)

    return model_path


def mock_messages(
    worker_manager_queue: "mp.Queue[bytes]",
    feature_store: FeatureStore,
    feature_store_root_dir: pathlib.Path,
    comm_channel_root_dir: pathlib.Path,
) -> None:
    """Mock event producer for triggering the inference pipeline"""
    feature_store_root_dir.mkdir(parents=True, exist_ok=True)
    comm_channel_root_dir.mkdir(parents=True, exist_ok=True)

    model_path = persist_model_file(feature_store_root_dir.parent / "model_original.pt")
    model_bytes = model_path.read_bytes()
    model_key = str(feature_store_root_dir / "model_fs.pt")

    feature_store[model_key] = model_bytes

    iteration_number = 0

    while True:
        iteration_number += 1
        time.sleep(1)
        # 1. for demo, ignore upstream and just put stuff into downstream
        # 2. for demo, only one downstream but we'd normally have to filter
        #       msg content and send to the correct downstream (worker) queue
        # timestamp = time.time_ns()
        # mock_channel = test_path / f"brainstorm-{timestamp}.txt"
        # mock_channel.touch()

        # thread - just look for key (wait for keys)
        # call checkpoint, try to get non-persistent key, it blocks
        # working set size > 1 has side-effects
        # only incurs cost when working set size has been exceeded

        channel_key = comm_channel_root_dir / f"{iteration_number}/channel.txt"
        callback_channel = FileSystemCommChannel(pathlib.Path(channel_key))

        input_path = feature_store_root_dir / f"{iteration_number}/input.pt"
        output_path = feature_store_root_dir / f"{iteration_number}/output.pt"

        input_key = str(input_path)
        output_key = str(output_path)

        buffer = io.BytesIO()
        tensor = torch.randn((1, 2), dtype=torch.float32)
        torch.save(tensor, buffer)
        feature_store[input_key] = buffer.getvalue()

        message_tensor_output_key = MessageHandler.build_tensor_key(output_key)
        message_tensor_input_key = MessageHandler.build_tensor_key(input_key)
        message_model_key = MessageHandler.build_model_key(model_key)

        request = MessageHandler.build_request(
            reply_channel=callback_channel.descriptor,
            model=message_model_key,
            inputs=[message_tensor_input_key],
            outputs=[message_tensor_output_key],
            custom_attributes=None,
        )
        request_bytes = MessageHandler.serialize_request(request)
        worker_manager_queue.put(request_bytes)


@pytest.fixture
def prepare_environment(test_dir: str) -> pathlib.Path:
    """Cleanup prior outputs to run demo repeatedly"""
    path = pathlib.Path(f"{test_dir}/workermanager.log")
    logging.basicConfig(filename=path.absolute(), level=logging.DEBUG)
    return path


def test_worker_manager(prepare_environment: pathlib.Path) -> None:
    """Test the worker manager"""

    test_path = prepare_environment
    fs_path = test_path / "feature_store"
    comm_path = test_path / "comm_store"

    config_loader = EnvironmentConfigLoader()
    integrated_worker = IntegratedTorchWorker()

    worker_manager = WorkerManager(
        config_loader,
        integrated_worker,
        as_service=True,
        cooldown=10,
        comm_channel_type=FileSystemCommChannel,
    )

    # create a mock client application to populate the request queue
    msg_pump = mp.Process(
        target=mock_messages,
        args=(
            config_loader.get_queue(),
            config_loader.get_feature_store(),
            fs_path,
            comm_path,
        ),
    )
    msg_pump.start()

    # # create a process to process commands
    process = mp.Process(target=worker_manager.execute)
    process.start()
    process.join(timeout=5)
    process.kill()
    msg_pump.kill()
