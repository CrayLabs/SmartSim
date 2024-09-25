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
import logging
import os
import pathlib
import subprocess as sp
import sys
import time
import typing as t
from queue import Empty

import numpy as np
import pytest

import conftest

pytest.importorskip("dragon")


# isort: off
import dragon
import multiprocessing as mp

import torch

# isort: on

from dragon import fli
from dragon.data.ddict.ddict import DDict
from dragon.managed_memory import MemoryAlloc

from smartsim._core.mli.comm.channel.dragon_channel import (
    DragonCommChannel,
    create_local,
)
from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
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
from smartsim._core.mli.infrastructure.worker.torch_worker import TorchWorker
from smartsim.log import get_logger

logger = get_logger(__name__)
mock_msg_pump_path = pathlib.Path(__file__).parent / "utils" / "msg_pump.py"
_MsgPumpFactory = t.Callable[[conftest.MsgPumpRequest], sp.Popen]

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


try:
    mp.set_start_method("dragon")
except Exception:
    pass


@pytest.mark.parametrize("num_iterations", [4])
def test_request_dispatcher(
    msg_pump_factory: _MsgPumpFactory, num_iterations: int
) -> None:
    """Test the request dispatcher batching and queueing system

    This also includes setting a queue to disposable, checking that it is no
    longer referenced by the dispatcher.
    """

    to_worker_channel = create_local()
    to_worker_fli = fli.FLInterface(main_ch=to_worker_channel, manager_ch=None)
    to_worker_fli_comm_ch = DragonFLIChannel(to_worker_fli, sender_supplied=True)

    ddict = DDict(1, 2, 4 * 1024**2)
    backbone_fs = BackboneFeatureStore(ddict, allow_reserved_writes=True)

    # NOTE: env vars should be set prior to instantiating EnvironmentConfigLoader
    # or test environment may be unable to send messages w/queue
    os.environ[BackboneFeatureStore.MLI_WORKER_QUEUE] = to_worker_fli_comm_ch.descriptor
    os.environ[BackboneFeatureStore.MLI_BACKBONE] = backbone_fs.descriptor

    config_loader = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=DragonCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_sender_supplied_descriptor,
    )

    request_dispatcher = RequestDispatcher(
        batch_timeout=0,
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
    pump_processes: t.List[sp.Popen] = []

    for i in range(num_iterations):
        batch: t.Optional[RequestBatch] = None
        mem_allocs = []
        tensors = []

        # NOTE: creating callbacks in test to avoid a local channel being torn
        # down when mock_messages terms but before the final response message is sent

        callback_channel = DragonCommChannel.from_local()

        request = conftest.MsgPumpRequest(
            backbone_fs.descriptor,
            worker_queue.descriptor,
            callback_channel.descriptor,
            i,
        )

        msg_pump = msg_pump_factory(request)
        pump_processes.append(msg_pump)

        time.sleep(1)

        for _ in range(200):
            try:
                request_dispatcher._on_iteration()
                batch = request_dispatcher.task_queue.get(timeout=0.1)
                break
            except Empty:
                continue
            except Exception as exc:
                raise exc

        assert batch is not None
        assert batch.has_valid_requests

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

        request_dispatcher._active_queues[model_key].make_disposable()
        assert request_dispatcher._active_queues[model_key].can_be_removed

        request_dispatcher._on_iteration()

        assert model_key not in request_dispatcher._active_queues
        assert model_key not in request_dispatcher._queues

        msg_pump.wait()

    for msg_pump in pump_processes:
        if msg_pump.returncode is not None:
            continue
        msg_pump.terminate()

    # Try to remove the dispatcher and free the memory
    del request_dispatcher
    gc.collect()
