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
import os
import pathlib
import subprocess as sp
import time
import typing as t
from queue import Empty

import numpy as np
import pytest

from . import conftest
from .utils import msg_pump

pytest.importorskip("dragon")


# isort: off
import dragon
import multiprocessing as mp

import torch

# isort: on

from dragon import fli
from dragon.data.ddict.ddict import DDict
from dragon.managed_memory import MemoryAlloc

from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
from smartsim._core.mli.comm.channel.dragon_util import create_local
from smartsim._core.mli.infrastructure.control.request_dispatcher import (
    RequestBatch,
    RequestDispatcher,
)
from smartsim._core.mli.infrastructure.control.worker_manager import (
    EnvironmentConfigLoader,
)
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.dragon_feature_store import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.feature_store import ModelKey, TensorKey
from smartsim._core.mli.infrastructure.worker.torch_worker import TorchWorker
from smartsim._core.mli.infrastructure.worker.worker import InferenceRequest, TensorMeta
from smartsim._core.mli.message_handler import MessageHandler
from smartsim.log import get_logger
from tests.dragon.utils.channel import FileSystemCommChannel

logger = get_logger(__name__)

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


try:
    mp.set_start_method("dragon")
except Exception:
    pass


@pytest.mark.parametrize("num_iterations", [4])
def test_request_dispatcher(
    num_iterations: int,
    the_storage: DDict,
    test_dir: str,
) -> None:
    """Test the request dispatcher batching and queueing system

    This also includes setting a queue to disposable, checking that it is no
    longer referenced by the dispatcher.
    """

    to_worker_channel = create_local()
    to_worker_fli = fli.FLInterface(main_ch=to_worker_channel, manager_ch=None)
    to_worker_fli_comm_ch = DragonFLIChannel(to_worker_fli)

    backbone_fs = BackboneFeatureStore(the_storage, allow_reserved_writes=True)

    # NOTE: env vars should be set prior to instantiating EnvironmentConfigLoader
    # or test environment may be unable to send messages w/queue
    os.environ[BackboneFeatureStore.MLI_WORKER_QUEUE] = to_worker_fli_comm_ch.descriptor
    os.environ[BackboneFeatureStore.MLI_BACKBONE] = backbone_fs.descriptor

    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=DragonCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_descriptor,
    )

    request_dispatcher = RequestDispatcher(
        batch_timeout=1000,
        batch_size=2,
        config_loader=config_loader,
        worker_type=TorchWorker,
        mem_pool_size=2 * 1024**2,
    )

    worker_queue = config_loader.get_queue()
    if worker_queue is None:
        logger.warning(
            "FLI input queue not loaded correctly from config_loader: "
            f"{config_loader._queue_descriptor}"
        )

    request_dispatcher._on_start()

    # put some messages into the work queue for the dispatcher to pickup
    channels = []
    processes = []
    for i in range(num_iterations):
        batch: t.Optional[RequestBatch] = None
        mem_allocs = []
        tensors = []

        # NOTE: creating callbacks in test to avoid a local channel being torn
        # down when mock_messages terms but before the final response message is sent

        callback_channel = DragonCommChannel.from_local()
        channels.append(callback_channel)

        process = conftest.function_as_dragon_proc(
            msg_pump.mock_messages,
            [
                worker_queue.descriptor,
                backbone_fs.descriptor,
                i,
                callback_channel.descriptor,
            ],
            [],
            [],
        )
        processes.append(process)
        process.start()
        assert process.returncode is None, "The message pump failed to start"
        # give dragon some time to populate the message queues
        for i in range(15):
            try:
                request_dispatcher._on_iteration()
                batch = request_dispatcher.task_queue.get(timeout=1.0)
                break
            except Empty:
                time.sleep(2)
                logger.warning(f"Task queue is empty on iteration {i}")
                continue
            except Exception as exc:
                logger.error(f"Task queue exception on iteration {i}")
                raise exc

        assert batch is not None
        assert batch.has_callbacks

        model_key = batch.model_id.key

        try:
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

            assert len(batch.callback_descriptors) == 2
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

        request_dispatcher._active_queues[model_key].make_disposable()
        assert request_dispatcher._active_queues[model_key].can_be_removed

        request_dispatcher._on_iteration()

        assert model_key not in request_dispatcher._active_queues
        assert model_key not in request_dispatcher._queues

    # Try to remove the dispatcher and free the memory
    del request_dispatcher
    gc.collect()


def test_request_batch(test_dir: str) -> None:
    """Test the RequestBatch.from_requests instantiates properly"""
    tensor_key = TensorKey(key="key", descriptor="desc1")
    tensor_key2 = TensorKey(key="key2", descriptor="desc1")
    output_key = TensorKey(key="key", descriptor="desc2")
    output_key2 = TensorKey(key="key2", descriptor="desc2")
    model_id1 = ModelKey(key="model key", descriptor="model desc")
    model_id2 = ModelKey(key="model key2", descriptor="model desc")
    tensor_desc = MessageHandler.build_tensor_descriptor("c", "float32", [1, 2])
    req_batch_model_id = ModelKey(key="req key", descriptor="desc")

    callback1 = FileSystemCommChannel(pathlib.Path(test_dir) / "callback1").descriptor
    callback2 = FileSystemCommChannel(pathlib.Path(test_dir) / "callback2").descriptor
    callback3 = FileSystemCommChannel(pathlib.Path(test_dir) / "callback3").descriptor

    request1 = InferenceRequest(
        model_key=model_id1,
        callback_desc=callback1,
        raw_inputs=[b"input data"],
        input_keys=[tensor_key],
        input_meta=[tensor_desc],
        output_keys=[output_key],
        raw_model=b"model",
        batch_size=0,
    )

    request2 = InferenceRequest(
        model_key=model_id2,
        callback_desc=callback2,
        raw_inputs=None,
        input_keys=None,
        input_meta=None,
        output_keys=[output_key, output_key2],
        raw_model=b"model",
        batch_size=0,
    )

    request3 = InferenceRequest(
        model_key=model_id2,
        callback_desc=callback3,
        raw_inputs=None,
        input_keys=[tensor_key, tensor_key2],
        input_meta=[tensor_desc],
        output_keys=None,
        raw_model=b"model",
        batch_size=0,
    )

    request_batch = RequestBatch.from_requests(
        [request1, request2, request3], req_batch_model_id
    )

    assert len(request_batch.callback_descriptors) == 3
    for callback in request_batch.callback_descriptors:
        assert isinstance(callback, str)
    assert len(request_batch.output_key_refs.keys()) == 2
    assert request_batch.has_callbacks
    assert request_batch.model_id == req_batch_model_id
    assert request_batch.inputs == None
    assert request_batch.raw_model == b"model"
    assert request_batch.raw_inputs == [[b"input data"]]
    assert request_batch.input_meta == [
        [TensorMeta([1, 2], "c", "float32")],
        [TensorMeta([1, 2], "c", "float32")],
    ]
    assert request_batch.input_keys == [
        [tensor_key],
        [tensor_key, tensor_key2],
    ]
    assert request_batch.output_key_refs == {
        callback1: [output_key],
        callback2: [output_key, output_key2],
    }


def test_request_batch_has_no_callbacks():
    """Verify that a request batch with no callbacks is correctly identified"""
    request1 = InferenceRequest()
    request2 = InferenceRequest()

    batch = RequestBatch.from_requests([request1, request2], ModelKey("model", "desc"))

    assert not batch.has_callbacks
