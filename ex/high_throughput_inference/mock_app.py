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
import numpy
import os
import time
import torch

from mpi4py import MPI
from smartsim._core.mli.infrastructure.storage.dragon_feature_store import (
    DragonFeatureStore,
)
from smartsim._core.utils.timings import PerfTimer

torch.set_num_interop_threads(16)
torch.set_num_threads(1)

logger = get_logger("App")
logger.info("Started app")
import typing as t
import numbers

from collections import OrderedDict
from smartsim._core.mli.comm.channel.dragonchannel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragonfli import DragonFLIChannel
from smartsim._core.mli.infrastructure.storage.featurestore import ReservedKeys
from smartsim._core.mli.message_handler import MessageHandler
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
    EventBroadcaster,
    EventPublisher,
    OnWriteFeatureStore,
)
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger, log_to_file

logger = get_logger("App", "DEBUG")

log_to_file("_smartsim.log", "debug")


_TIMING_DICT = OrderedDict[str, list[numbers.Number]]

CHECK_RESULTS_AND_MAKE_ALL_SLOWER = False

class ProtoClient:
    def __init__(self, timing_on: bool):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    def _attach_to_backbone(
        self, wait_timeout: float = 0, create_bb: bool = False
    ) -> BackboneFeatureStore:
        """Use the supplied environment variables to attach
        to a pre-existing backbone featurestore. Requires the
        environment to contain `_SMARTSIM_INFRA_BACKBONE`
        environment variable

        :returns: the attached backbone featurestore"""
        # todo: ensure this env var from config loader or constant
        descriptor = os.environ.get("_SMARTSIM_INFRA_BACKBONE", None)
        if descriptor is None:
            raise SmartSimError(
                "Missing required backbone configuration in environment"
            )

        backbone = t.cast(
            BackboneFeatureStore, BackboneFeatureStore.from_descriptor(descriptor)
        )
        backbone.wait_timeout = wait_timeout
        return backbone

    def _attach_to_worker_queue(self) -> None:
        """Wait until the backbone contains the worker queue configuration,
        then attach an FLI to the given worker queue"""
        configuration = self._backbone.wait_for(
            [ReservedKeys.MLI_WORKER_QUEUE], timeout=self._queue_timeout
        )

        descriptor = configuration.get(ReservedKeys.MLI_WORKER_QUEUE, None)
        if not descriptor:
            raise ValueError("Unable to locate worker queue using backbone")

        # self._to_worker_fli = DragonFLIChannel.from_descriptor(descriptor)
        return DragonFLIChannel.from_descriptor(descriptor)

    def _create_worker_channels(self) -> t.Tuple[DragonCommChannel, DragonCommChannel]:
        """Create channels to be used in the worker queue"""
        # self._from_worker_ch = Channel.make_process_local()
        _from_worker_ch = DragonCommChannel.from_local()
        # self._from_worker_ch_serialized = self._from_worker_ch.serialize()
        # self._to_worker_ch = Channel.make_process_local()
        _to_worker_ch = DragonCommChannel.from_local()

        return _from_worker_ch, _to_worker_ch

    def _create_publisher(self) -> EventPublisher:
        """Create an event publisher that will broadcast updates to
        other MLI components. This publisher

        :returns: the event publisher instance"""
        publisher: EventPublisher = EventBroadcaster(
            self._backbone, DragonCommChannel.from_descriptor
        )
        return publisher

    def __init__(self, timing_on: bool, wait_timeout: float = 0):
        """Initialize the client instance

        :param timing_on: Flag indicating if timing information should be written to file
        :param wait_timeout: Maximum wait time allowed to attach to the worker queue

        :raises: SmartSimError if unable to attach to a backbone featurestore"""
        self._queue_timeout = wait_timeout

        connect_to_infrastructure()
        # ddict_str = os.environ["_SMARTSIM_INFRA_BACKBONE"]
        # self._ddict = DDict.attach(ddict_str)
        # self._backbone_descriptor = DragonFeatureStore(self._ddict).descriptor
        self._backbone = self._attach_to_backbone(wait_timeout=wait_timeout)

        # # to_worker_fli_str = None
        # # while to_worker_fli_str is None:
        # #     try:
        # #         to_worker_fli_str = self._ddict["to_worker_fli"]
        # #         self._to_worker_fli = fli.FLInterface.attach(to_worker_fli_str)
        # #     except KeyError:
        # #         time.sleep(1)

        self._to_worker_fli = self._attach_to_worker_queue()

        # # # self._from_worker_ch = Channel.make_process_local()
        # # # self._from_worker_ch_serialized = self._from_worker_ch.serialize()
        # # # self._to_worker_ch = Channel.make_process_local()
        channels = self._create_worker_channels()
        self._from_worker_ch = channels[0]
        self._to_worker_ch = channels[1]

        self._publisher = self._create_publisher()

        self.perf_timer: PerfTimer = PerfTimer(debug=False, timing_on=timing_on, prefix=f"a{rank}_")
        self._start = None
        self._interm = None
        self._timings: _TIMING_DICT = OrderedDict()
        self._timing_on = timing_on

    def _add_label_to_timings(self, label: str):
        if label not in self._timings:
            self._timings[label] = []

    @staticmethod
    def _format_number(number: numbers.Number):
        return f"{number:0.4e}"

    def start_timings(self, batch_size: int):
        if self._timing_on:
            self._add_label_to_timings("batch_size")
            self._timings["batch_size"].append(batch_size)
            self._start = time.perf_counter()
            self._interm = time.perf_counter()

    def end_timings(self):
        if self._timing_on:
            self._add_label_to_timings("total_time")
            self._timings["total_time"].append(
                self._format_number(time.perf_counter() - self._start)
            )

    def measure_time(self, label: str):
        if self._timing_on:
            self._add_label_to_timings(label)
            self._timings[label].append(
                self._format_number(time.perf_counter() - self._interm)
            )
            self._interm = time.perf_counter()

    def print_timings(self, to_file: bool = False):
        print(" ".join(self._timings.keys()))
        value_array = numpy.array(
            [value for value in self._timings.values()], dtype=float
        )
        value_array = numpy.transpose(value_array)
        for i in range(value_array.shape[0]):
            print(" ".join(self._format_number(value) for value in value_array[i]))
        if to_file:
            numpy.save("timings.npy", value_array)
            numpy.savetxt("timings.txt", value_array)

    def run_model(self, model: bytes | str, batch: torch.Tensor):
        tensors = [batch.numpy()]
        self.perf_timer.start_timings("batch_size", batch.shape[0])
        built_tensor_desc = MessageHandler.build_tensor_descriptor(
            "c", "float32", list(batch.shape)
        )
        self.perf_timer.measure_time("build_tensor_descriptor")
        if isinstance(model, str):
            model_arg = MessageHandler.build_feature_store_key(
                model, self._backbone.descriptor
            )
        else:
            model_arg = MessageHandler.build_model(model, "resnet-50", "1.0")
        request = MessageHandler.build_request(
            reply_channel=self._from_worker_ch.descriptor,
            model=model_arg,
            inputs=[built_tensor_desc],
            outputs=[],
            output_descriptors=[],
            custom_attributes=None,
        )
        self.perf_timer.measure_time("build_request")
        request_bytes = MessageHandler.serialize_request(request)
        self.perf_timer.measure_time("serialize_request")

        with self._to_worker_fli._channel.sendh(
            timeout=None, stream_channel=self._to_worker_ch.channel
        ) as to_sendh:
            to_sendh.send_bytes(request_bytes)
            self.perf_timer.measure_time("send_request")
            for t in tensors:
                to_sendh.send_bytes(t.tobytes())  # TODO NOT FAST ENOUGH!!!
                # to_sendh.send_bytes(bytes(t.data))
        logger.info(f"Message size: {len(request_bytes)} bytes")

        self.perf_timer.measure_time("send_tensors")
        with self._from_worker_ch.channel.recvh(timeout=None) as from_recvh:
            resp = from_recvh.recv_bytes(timeout=None)
            self.perf_timer.measure_time("receive_response")
            response = MessageHandler.deserialize_response(resp)
            self.perf_timer.measure_time("deserialize_response")
            # list of data blobs? recv depending on the len(response.result.descriptors)?
            data_blob: bytes = from_recvh.recv_bytes(timeout=None)
            self.perf_timer.measure_time("receive_tensor")
            result = torch.from_numpy(
                numpy.frombuffer(
                    data_blob,
                    dtype=str(response.result.descriptors[0].dataType),
                )
            )
            self.perf_timer.measure_time("deserialize_tensor")

        self.perf_timer.end_timings()
        return result

    def set_model(self, key: str, model: bytes):
        # todo: incorrect usage of backbone here to store
        # user models? are we using the backbone if they do NOT
        # have a feature store of their own?
        self._backbone[key] = model

        # notify components of a change in the data at this key
        event = OnWriteFeatureStore(self._backbone.descriptor, key)
        self._publisher.send(event)



class ResNetWrapper:
    def __init__(self, name: str, model: str):
        self._model = torch.jit.load(model)
        self._name = name
        buffer = io.BytesIO()
        scripted = torch.jit.trace(self._model, self.get_batch())
        torch.jit.save(scripted, buffer)
        self._serialized_model = buffer.getvalue()

    def get_batch(self, batch_size: int = 32):
        return torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)

    @property
    def model(self):
        return self._serialized_model

    @property
    def name(self):
        return self._name

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mock application")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--log_max_batchsize", default=8, type=int)
    args = parser.parse_args()

    resnet = ResNetWrapper("resnet50", f"resnet50.{args.device}.pt")

    client = ProtoClient(timing_on=True, wait_timeout=0)
    # client.set_model(resnet.name, resnet.model)

    if CHECK_RESULTS_AND_MAKE_ALL_SLOWER:
        # TODO: adapt to non-Nvidia devices
        torch_device = args.device.replace("gpu", "cuda")
        pt_model = torch.jit.load(io.BytesIO(initial_bytes=(resnet.model))).to(torch_device)

    TOTAL_ITERATIONS = 100

    for log2_bsize in range(args.log_max_batchsize+1):
        b_size: int = 2**log2_bsize
        logger.info(f"Batch size: {b_size}")
        for iteration_number in range(TOTAL_ITERATIONS + int(b_size==1)):
            logger.info(f"Iteration: {iteration_number}")
            sample_batch = resnet.get_batch(b_size)
            remote_result = client.run_model(resnet.name, sample_batch)
            logger.info(client.perf_timer.get_last("total_time"))
            if CHECK_RESULTS_AND_MAKE_ALL_SLOWER:
                local_res = pt_model(sample_batch.to(torch_device))
                err_norm = torch.linalg.vector_norm(torch.flatten(remote_result).to(torch_device)-torch.flatten(local_res), ord=1).cpu()
                res_norm = torch.linalg.vector_norm(remote_result, ord=1).item()
                local_res_norm = torch.linalg.vector_norm(local_res, ord=1).item()
                logger.info(f"Avg norm of error {err_norm.item()/b_size} compared to result norm of {res_norm/b_size}:{local_res_norm/b_size}")
                torch.cuda.synchronize()

    client.perf_timer.print_timings(to_file=True)
