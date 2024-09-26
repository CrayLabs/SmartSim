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

# isort: off
import dragon
from dragon import fli
from dragon.channels import Channel
import dragon.channels
from dragon.data.ddict.ddict import DDict
from dragon.globalservices.api_setup import connect_to_infrastructure
from dragon.utils import b64decode, b64encode

# isort: on

import argparse
import io
import os
import time
import typing as t
import warnings

import numpy
import torch
from mpi4py import MPI

from smartsim._core.mli.infrastructure.storage.dragon_feature_store import (
    DragonFeatureStore,
)
from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel, create_local
from smartsim._core.mli.message_handler import MessageHandler
from smartsim._core.utils.timings import PerfTimer
from smartsim.log import get_logger

torch.set_num_interop_threads(16)
torch.set_num_threads(1)

logger = get_logger("App")

warnings.filterwarnings("ignore", "\*The given NumPy array is not writable\*")


class ProtoClient:
    def __init__(self, timing_on: bool):
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        connect_to_infrastructure()
        ddict_str = os.environ["_SMARTSIM_INFRA_BACKBONE"]
        self._ddict = DDict.attach(ddict_str)
        self._backbone_descriptor = DragonFeatureStore(self._ddict).descriptor
        to_worker_fli_str = None
        while to_worker_fli_str is None:
            try:
                to_worker_fli_str = self._ddict["to_worker_fli"]
                self._to_worker_fli = fli.FLInterface.attach(to_worker_fli_str)
            except KeyError:
                time.sleep(1)
        self._from_worker_ch = DragonCommChannel(Channel.make_process_local())
        self._from_worker_ch_serialized = self._from_worker_ch.descriptor_string
        self._to_worker_ch = Channel.make_process_local()

        self.perf_timer: PerfTimer = PerfTimer(
            debug=False, timing_on=timing_on, prefix=f"a{self._rank}_"
        )
        self._num_its: int = 0

    def run_model(self, model: t.Union[bytes, str], batch: torch.Tensor):
        tensors = [batch.numpy()]
        self.perf_timer.start_timings("batch_size", batch.shape[0])
        built_tensor_desc = MessageHandler.build_tensor_descriptor(
            "c", "float32", list(batch.shape)
        )
        self.perf_timer.measure_time("build_tensor_descriptor")
        if isinstance(model, str):
            model_arg = MessageHandler.build_model_key(model, self._backbone_descriptor)
        else:
            model_arg = MessageHandler.build_model(model, "resnet-50", "1.0")
        request = MessageHandler.build_request(
            reply_channel=self._from_worker_ch_serialized,
            model=model_arg,
            inputs=[built_tensor_desc],
            outputs=[],
            output_descriptors=[],
            custom_attributes=None,
        )
        self.perf_timer.measure_time("build_request")
        request_bytes = MessageHandler.serialize_request(request)
        self.perf_timer.measure_time("serialize_request")
        with self._to_worker_fli.sendh(
            timeout=None, stream_channel=self._to_worker_ch
        ) as to_sendh:
            to_sendh.send_bytes(request_bytes)
            self.perf_timer.measure_time("send_request")
            for tensor in tensors:
                to_sendh.send_bytes(tensor.tobytes())  # TODO NOT FAST ENOUGH!!!
        self.perf_timer.measure_time("send_tensors")

        resp = self._from_worker_ch.recv(timeout=None)
        self.perf_timer.measure_time("receive_response")
        response = MessageHandler.deserialize_response(resp[0])
        self.perf_timer.measure_time("deserialize_response")
        # list of data blobs? recv depending on the len(response.result.descriptors)?
        if len(resp) > 1:
            data_blob = resp[1]
        else:
            data_blob: bytes = self._from_worker_ch.recv(timeout=None)[0]
        self.perf_timer.measure_time("receive_tensor")
        result = torch.from_numpy(
            numpy.frombuffer(
                data_blob,
                dtype=str(response.result.descriptors[0].dataType),
            )
        )
        self.perf_timer.measure_time("deserialize_tensor")

        self.perf_timer.end_timings()
        self._num_its += 1
        self._comm.Barrier()
        return result

    def set_model(self, key: str, model: bytes):
        self._ddict[key] = model


class ResNetWrapper:
    def __init__(self, name: str, model: str):
        self._model = None  # torch.jit.load(model)
        self._name = name

        with open(model, "rb") as model_file:
            buffer = io.BytesIO(model_file.read())
        self._serialized_model = buffer.getvalue()

    def get_batch(self, batch_size: int = 32):
        return torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)

    @property
    def model(self):
        return self._serialized_model

    @property
    def name(self):
        return self._name


def log(msg: str, rank: int) -> None:
    if rank == 0:
        logger.info(msg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mock application")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--log_max_batchsize", default=8, type=int)
    args = parser.parse_args()

    resnet = ResNetWrapper("resnet50", f"resnet50.{args.device}.pt")

    client = ProtoClient(timing_on=True)
    if client._rank == 0:
        client.set_model(resnet.name, resnet.model)

    MPI.COMM_WORLD.Barrier()

    TOTAL_ITERATIONS = 100

    for log2_bsize in range(args.log_max_batchsize, args.log_max_batchsize + 1):
        b_size: int = 2**log2_bsize
        log(f"Batch size: {b_size}", client._rank)
        for iteration_number in range(TOTAL_ITERATIONS):
            sample_batch = resnet.get_batch(b_size)
            remote_result = client.run_model(resnet.name, sample_batch)
            log(
                f"Completed iteration: {iteration_number} in {client.perf_timer.get_last('total_time')} seconds",
                client._rank,
            )

    client.perf_timer.print_timings(to_file=True, to_stdout=client._rank == 0)
