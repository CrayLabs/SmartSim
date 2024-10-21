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
import pathlib
import time

import pytest

torch = pytest.importorskip("torch")
dragon = pytest.importorskip("dragon")

import multiprocessing as mp

try:
    mp.set_start_method("dragon")
except Exception:
    pass

import os

import torch.nn as nn
from dragon import fli

from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
from smartsim._core.mli.comm.channel.dragon_util import create_local
from smartsim._core.mli.infrastructure.control.worker_manager import (
    EnvironmentConfigLoader,
    WorkerManager,
)
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.dragon_feature_store import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.dragon_util import create_ddict
from smartsim._core.mli.infrastructure.worker.torch_worker import TorchWorker
from smartsim._core.mli.message_handler import MessageHandler
from smartsim.log import get_logger

from .utils.channel import FileSystemCommChannel

logger = get_logger(__name__)
# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


class MiniModel(nn.Module):
    """A torch model that can be executed by the default torch worker"""

    def __init__(self):
        """Initialize the model."""
        super().__init__()

        self._name = "mini-model"
        self._net = torch.nn.Linear(2, 1)

    def forward(self, input):
        """Execute a forward pass."""
        return self._net(input)

    @property
    def bytes(self) -> bytes:
        """Retrieve the serialized model

        :returns: The byte stream of the model file
        """
        buffer = io.BytesIO()
        scripted = torch.jit.trace(self._net, self.get_batch())
        torch.jit.save(scripted, buffer)
        return buffer.getvalue()

    @classmethod
    def get_batch(cls) -> "torch.Tensor":
        """Generate a single batch of data with the correct
        shape for inference.

        :returns: The batch as a torch tensor
        """
        return torch.randn((100, 2), dtype=torch.float32)


def create_model(model_path: pathlib.Path) -> pathlib.Path:
    """Create a simple torch model and persist to disk for
    testing purposes.

    :param model_path: The path to the torch model file
    """
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)

    model_path.unlink(missing_ok=True)

    mini_model = MiniModel()
    torch.save(mini_model, model_path)

    return model_path


def load_model() -> bytes:
    """Create a simple torch model in memory for testing."""
    mini_model = MiniModel()
    return mini_model.bytes


def mock_messages(
    feature_store_root_dir: pathlib.Path,
    comm_channel_root_dir: pathlib.Path,
    kill_queue: mp.Queue,
) -> None:
    """Mock event producer for triggering the inference pipeline.

    :param feature_store_root_dir: Path to a directory where a
    FileSystemFeatureStore can read & write results
    :param comm_channel_root_dir: Path to a directory where a
    FileSystemCommChannel can read & write messages
    :param kill_queue: Queue used by unit test to stop mock_message process
    """
    feature_store_root_dir.mkdir(parents=True, exist_ok=True)
    comm_channel_root_dir.mkdir(parents=True, exist_ok=True)

    iteration_number = 0

    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=FileSystemCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_descriptor,
    )
    backbone = config_loader.get_backbone()

    worker_queue = config_loader.get_queue()
    if worker_queue is None:
        queue_desc = config_loader._queue_descriptor
        logger.warn(
            f"FLI input queue not loaded correctly from config_loader: {queue_desc}"
        )

    model_key = "mini-model"
    model_bytes = load_model()
    backbone[model_key] = model_bytes

    while True:
        if not kill_queue.empty():
            return
        iteration_number += 1
        time.sleep(1)

        channel_key = comm_channel_root_dir / f"{iteration_number}/channel.txt"
        callback_channel = FileSystemCommChannel(pathlib.Path(channel_key))

        batch = MiniModel.get_batch()
        shape = batch.shape
        batch_bytes = batch.numpy().tobytes()

        logger.debug(f"Model content: {backbone[model_key][:20]}")

        input_descriptor = MessageHandler.build_tensor_descriptor(
            "f", "float32", list(shape)
        )

        # The first request is always the metadata...
        request = MessageHandler.build_request(
            reply_channel=callback_channel.descriptor,
            model=MessageHandler.build_model(model_bytes, "mini-model", "1.0"),
            inputs=[input_descriptor],
            outputs=[],
            output_descriptors=[],
            custom_attributes=None,
        )
        request_bytes = MessageHandler.serialize_request(request)
        fli: DragonFLIChannel = worker_queue

        with fli._fli.sendh(timeout=None, stream_channel=fli._channel) as sendh:
            sendh.send_bytes(request_bytes)
            sendh.send_bytes(batch_bytes)

        logger.info("published message")

        if iteration_number > 5:
            return


def mock_mli_infrastructure_mgr() -> None:
    """Create resources normally instanatiated by the infrastructure
    management portion of the DragonBackend.
    """
    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=FileSystemCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_descriptor,
    )

    integrated_worker = TorchWorker

    worker_manager = WorkerManager(
        config_loader,
        integrated_worker,
        as_service=True,
        cooldown=10,
        device="cpu",
        dispatcher_queue=mp.Queue(maxsize=0),
    )
    worker_manager.execute()


@pytest.fixture
def prepare_environment(test_dir: str) -> pathlib.Path:
    """Cleanup prior outputs to run demo repeatedly.

    :param test_dir: the directory to prepare
    :returns: The path to the log file
    """
    path = pathlib.Path(f"{test_dir}/workermanager.log")
    logging.basicConfig(filename=path.absolute(), level=logging.DEBUG)
    return path


def test_worker_manager(prepare_environment: pathlib.Path) -> None:
    """Test the worker manager.

    :param prepare_environment: Pass this fixture to configure
    global resources before the worker manager executes
    """

    test_path = prepare_environment
    fs_path = test_path / "feature_store"
    comm_path = test_path / "comm_store"

    mgr_per_node = 1
    num_nodes = 2
    mem_per_node = 128 * 1024**2

    storage = create_ddict(num_nodes, mgr_per_node, mem_per_node)
    backbone = BackboneFeatureStore(storage, allow_reserved_writes=True)

    to_worker_channel = create_local()
    to_worker_fli = fli.FLInterface(main_ch=to_worker_channel, manager_ch=None)

    to_worker_fli_comm_channel = DragonFLIChannel(to_worker_fli)

    # NOTE: env vars must be set prior to instantiating EnvironmentConfigLoader
    # or test environment may be unable to send messages w/queue
    os.environ[BackboneFeatureStore.MLI_WORKER_QUEUE] = (
        to_worker_fli_comm_channel.descriptor
    )
    os.environ[BackboneFeatureStore.MLI_BACKBONE] = backbone.descriptor

    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=FileSystemCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_descriptor,
    )
    integrated_worker_type = TorchWorker

    worker_manager = WorkerManager(
        config_loader,
        integrated_worker_type,
        as_service=True,
        cooldown=5,
        device="cpu",
        dispatcher_queue=mp.Queue(maxsize=0),
    )

    worker_queue = config_loader.get_queue()
    if worker_queue is None:
        logger.warn(
            f"FLI input queue not loaded correctly from config_loader: {config_loader._queue_descriptor}"
        )
    backbone.worker_queue = to_worker_fli_comm_channel.descriptor

    # create a mock client application to populate the request queue
    kill_queue = mp.Queue()
    msg_pump = mp.Process(
        target=mock_messages,
        args=(fs_path, comm_path, kill_queue),
    )
    msg_pump.start()

    # create a process to execute commands
    process = mp.Process(target=mock_mli_infrastructure_mgr)

    # let it send some messages before starting the worker manager
    msg_pump.join(timeout=5)
    process.start()
    msg_pump.join(timeout=5)
    kill_queue.put_nowait("kill!")
    process.join(timeout=5)
    msg_pump.kill()
    process.kill()
