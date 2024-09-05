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

import gc
import io
import logging
import pathlib
import socket
import time
import typing as t
from queue import Empty

import numpy as np
import pytest

torch = pytest.importorskip("torch")
dragon = pytest.importorskip("dragon")

import base64
import multiprocessing as mp

try:
    mp.set_start_method("dragon")
except Exception:
    pass

import os

import dragon.channels as dch
import dragon.infrastructure.policy as dragon_policy
import dragon.infrastructure.process_desc as dragon_process_desc
import dragon.native.process as dragon_process
from dragon import fli
from dragon.channels import Channel
from dragon.data.ddict.ddict import DDict
from dragon.managed_memory import MemoryAlloc, MemoryPool
from dragon.mpbridge.queues import DragonQueue

from smartsim._core.entrypoints.service import Service
from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
from smartsim._core.mli.infrastructure.control.request_dispatcher import (
    RequestBatch,
    RequestDispatcher,
)
from smartsim._core.mli.infrastructure.control.worker_manager import (
    EnvironmentConfigLoader,
)
from smartsim._core.mli.infrastructure.storage.dragon_feature_store import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.feature_store import FeatureStore
from smartsim._core.mli.infrastructure.worker.torch_worker import TorchWorker
from smartsim._core.mli.message_handler import MessageHandler
from smartsim.log import get_logger

from .feature_store import FileSystemFeatureStore
from .utils.channel import FileSystemCommChannel

logger = get_logger(__name__)
# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


def persist_model_file(model_path: pathlib.Path) -> pathlib.Path:
    """Create a simple torch model and persist to disk for
    testing purposes.

    TODO: remove once unit tests are in place"""
    # test_path = pathlib.Path(work_dir)
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)

    model_path.unlink(missing_ok=True)

    model = torch.nn.Linear(2, 1)
    torch.save(model, model_path)

    return model_path


def mock_messages(
    request_dispatcher_queue: DragonFLIChannel,
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

    for iteration_number in range(2):

        channel = Channel.make_process_local()
        callback_channel = DragonCommChannel(channel)

        input_path = feature_store_root_dir / f"{iteration_number}/input.pt"
        output_path = feature_store_root_dir / f"{iteration_number}/output.pt"

        input_key = str(input_path)
        output_key = str(output_path)

        tensor = (
            (iteration_number + 1) * torch.ones((1, 2), dtype=torch.float32)
        ).numpy()
        fsd = feature_store.descriptor

        tensor_desc = MessageHandler.build_tensor_descriptor(
            "c", "float32", list(tensor.shape)
        )

        message_tensor_output_key = MessageHandler.build_feature_store_key(
            output_key, fsd
        )
        message_tensor_input_key = MessageHandler.build_feature_store_key(
            input_key, fsd
        )
        message_model_key = MessageHandler.build_feature_store_key(model_key, fsd)

        request = MessageHandler.build_request(
            reply_channel=base64.b64encode(channel.serialize()).decode("utf-8"),
            model=message_model_key,
            inputs=[tensor_desc],
            outputs=[message_tensor_output_key],
            output_descriptors=[],
            custom_attributes=None,
        )
        request_bytes = MessageHandler.serialize_request(request)
        with request_dispatcher_queue._fli.sendh(
            timeout=None, stream_channel=request_dispatcher_queue._channel
        ) as sendh:
            sendh.send_bytes(request_bytes)
            sendh.send_bytes(tensor.tobytes())
        time.sleep(1)


@pytest.fixture
def prepare_environment(test_dir: str) -> pathlib.Path:
    """Cleanup prior outputs to run demo repeatedly"""
    path = pathlib.Path(f"{test_dir}/workermanager.log")
    logging.basicConfig(filename=path.absolute(), level=logging.DEBUG)
    return path


def service_as_dragon_proc(
    service: Service, cpu_affinity: list[int], gpu_affinity: list[int]
) -> dragon_process.Process:

    options = dragon_process_desc.ProcessOptions(make_inf_channels=True)
    local_policy = dragon_policy.Policy(
        placement=dragon_policy.Policy.Placement.HOST_NAME,
        host_name=socket.gethostname(),
        cpu_affinity=cpu_affinity,
        gpu_affinity=gpu_affinity,
    )
    return dragon_process.Process(
        target=service.execute,
        args=[],
        cwd=os.getcwd(),
        policy=local_policy,
        options=options,
        stderr=dragon_process.Popen.STDOUT,
        stdout=dragon_process.Popen.STDOUT,
    )


def test_request_dispatcher(prepare_environment: pathlib.Path) -> None:
    """Test the request dispatcher batching and queueing system

    This also includes setting a queue to disposable, checking that it is no
    longer referenced by the dispatcher.
    """

    test_path = prepare_environment
    fs_path = test_path / "feature_store"
    comm_path = test_path / "comm_store"

    to_worker_channel = dch.Channel.make_process_local()
    to_worker_fli = fli.FLInterface(main_ch=to_worker_channel, manager_ch=None)
    to_worker_fli_serialized = to_worker_fli.serialize()

    # NOTE: env vars should be set prior to instantiating EnvironmentConfigLoader
    # or test environment may be unable to send messages w/queue
    descriptor = base64.b64encode(to_worker_fli_serialized).decode("utf-8")
    os.environ["_SMARTSIM_REQUEST_QUEUE"] = descriptor

    ddict = DDict(1, 2, 4 * 1024**2)
    dragon_fs = DragonFeatureStore(ddict)

    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=DragonCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_descriptor,
    )
    integrated_worker_type = TorchWorker

    request_dispatcher = RequestDispatcher(
        batch_timeout=0,
        batch_size=2,
        config_loader=config_loader,
        worker_type=integrated_worker_type,
        mem_pool_size=2 * 1024**2,
    )

    worker_queue = config_loader.get_queue()
    if worker_queue is None:
        logger.warn(
            "FLI input queue not loaded correctly from config_loader: "
            f"{config_loader._queue_descriptor}"
        )

    request_dispatcher._on_start()

    for _ in range(2):
        batch: t.Optional[RequestBatch] = None
        mem_allocs = []
        tensors = []
        fs_path = test_path / f"feature_store"
        comm_path = test_path / f"comm_store"
        model_key = str(fs_path / "model_fs.pt")

        # create a mock client application to populate the request queue
        msg_pump = mp.Process(
            target=mock_messages,
            args=(
                worker_queue,
                dragon_fs,
                fs_path,
                comm_path,
            ),
        )

        msg_pump.start()

        time.sleep(1)

        for attempts in range(15):
            try:
                request_dispatcher._on_iteration()
                batch = request_dispatcher.task_queue.get(timeout=1)
                break
            except Empty:
                continue
            except Exception as exc:
                raise exc

        try:
            assert batch is not None
            assert batch.has_valid_requests

            transform_result = batch.inputs
            for transformed, dims, dtype in zip(
                transform_result.transformed,
                transform_result.dims,
                transform_result.dtypes,
            ):
                mem_alloc = MemoryAlloc.attach(transformed)
                mem_allocs.append(mem_alloc)
                itemsize = np.empty((1), dtype=dtype).itemsize
                tensors.append(
                    torch.from_numpy(
                        np.frombuffer(
                            mem_alloc.get_memview()[0 : np.prod(dims) * itemsize],
                            dtype=dtype,
                        ).reshape(dims)
                    )
                )

            assert len(batch.requests) == 2
            assert batch.model_id.key == model_key
            assert model_key in request_dispatcher._queues
            assert model_key in request_dispatcher._active_queues
            assert len(request_dispatcher._queues[model_key]) == 1
            assert request_dispatcher._queues[model_key][0].empty()
            assert request_dispatcher._queues[model_key][0].model_id.key == model_key
            assert len(tensors) == 1
            assert tensors[0].shape == torch.Size([2, 2])

            for tensor in tensors:
                for sample_idx in range(tensor.shape[0]):
                    tensor_in = tensor[sample_idx]
                    tensor_out = (sample_idx + 1) * torch.ones(
                        (2,), dtype=torch.float32
                    )
                    assert torch.equal(tensor_in, tensor_out)

        except Exception as exc:
            raise exc
        finally:
            for mem_alloc in mem_allocs:
                mem_alloc.free()

            msg_pump.kill()

        request_dispatcher._active_queues[model_key].make_disposable()
        assert request_dispatcher._active_queues[model_key].can_be_removed

        request_dispatcher._on_iteration()

        assert model_key not in request_dispatcher._active_queues
        assert model_key not in request_dispatcher._queues

    # Try to remove the dispatcher and free the memory
    del request_dispatcher
    gc.collect()
