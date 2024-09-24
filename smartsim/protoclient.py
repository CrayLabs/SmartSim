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
# pylint: disable=unused-import,import-error
import dragon
from dragon import fli
import dragon.channels
from dragon.globalservices.api_setup import connect_to_infrastructure

# isort: on
# pylint: enable=unused-import,import-error

import numbers
import os
import time
import typing as t
from collections import OrderedDict

import numpy
import torch

from smartsim._core.mli.comm.channel.dragon_channel import (
    create_local,
    DragonCommChannel,
)
from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
    EventBroadcaster,
    EventProducer,
    OnWriteFeatureStore,
)
from smartsim._core.mli.message_handler import MessageHandler
from smartsim._core.utils.timings import PerfTimer
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger


try:
    from mpi4py import MPI
except Exception:
    MPI = None
    print("Unable to import `mpi4py` package")

_TimingDict = OrderedDict[str, list[str]]


logger = get_logger("App")
logger.info("Started app")
CHECK_RESULTS_AND_MAKE_ALL_SLOWER = False


class ProtoClient:
    """Proof of concept implementation of a client enabling user applications
    to interact with MLI resources."""

    _DEFAULT_BACKBONE_TIMEOUT = 30.0
    """A default timeout period applied to connection attempts with the
    backbone feature store."""

    _DEFAULT_WORK_QUEUE_SIZE = 500
    """A default number of events to be buffered in the work queue before
    triggering QueueFull exceptions."""

    @staticmethod
    def _attach_to_backbone(wait_timeout: float = 0) -> BackboneFeatureStore:
        """Use the supplied environment variables to attach
        to a pre-existing backbone featurestore. Requires the
        environment to contain `_SMARTSIM_INFRA_BACKBONE`
        environment variable

        :returns: the attached backbone featurestore"""
        # todo: ensure this env var from config loader or constant
        descriptor = os.environ.get(BackboneFeatureStore.MLI_BACKBONE, None)
        if descriptor is None:
            raise SmartSimError(
                "Missing required backbone configuration in environment: "
                f"{BackboneFeatureStore.MLI_BACKBONE}"
            )

        backbone = t.cast(
            BackboneFeatureStore, BackboneFeatureStore.from_descriptor(descriptor)
        )
        backbone.wait_timeout = wait_timeout
        return backbone

    def _attach_to_worker_queue(self) -> DragonFLIChannel:
        """Wait until the backbone contains the worker queue configuration,
        then attach an FLI to the given worker queue"""

        descriptor = ""
        try:
            # NOTE: without wait_for, this MUST be in the backbone....
            config = self._backbone.wait_for(
                [BackboneFeatureStore.MLI_WORKER_QUEUE], self.backbone_timeout
            )
            descriptor = str(config[BackboneFeatureStore.MLI_WORKER_QUEUE])
        except Exception as ex:
            logger.info(
                f"Unable to rerieve {BackboneFeatureStore.MLI_WORKER_QUEUE} "
                "to attach to the worker queue."
            )
            raise ValueError("Unable to locate worker queue using backbone") from ex

        return DragonFLIChannel.from_descriptor(descriptor)

    @classmethod
    def _create_worker_channels(
        cls,
    ) -> t.Tuple[dragon.channels.Channel, dragon.channels.Channel]:
        """Create channels to be used for communication to and from the worker queue.

        :returns: A tuple containing the native from and to Channels as (from_channel, to_channel).
        """

        _from_worker_ch_raw = create_local(cls._DEFAULT_WORK_QUEUE_SIZE)
        _to_worker_ch_raw = create_local(cls._DEFAULT_WORK_QUEUE_SIZE)

        return _from_worker_ch_raw, _to_worker_ch_raw

    def _create_broadcaster(self) -> EventProducer:
        """Create an event publisher that will broadcast updates to
        other MLI components. This publisher

        :returns: the event publisher instance"""
        broadcaster = EventBroadcaster(
            self._backbone, DragonCommChannel.from_descriptor
        )
        return broadcaster

    def __init__(self, timing_on: bool, wait_timeout: float = 0) -> None:
        """Initialize the client instance

        :param timing_on: Flag indicating if timing information should be
        written to file
        :param wait_timeout: Maximum wait time (in seconds) allowed to attach to the
        worker queue

        :raises: SmartSimError if unable to attach to a backbone featurestore"""
        # todo: determine a way to make this work in tests.
        #  - consider catching the import exception and defaulting rank to 0
        if MPI is not None:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        else:
            rank: int = 0

        self._backbone_timeout = wait_timeout

        connect_to_infrastructure()

        self._backbone = self._attach_to_backbone(wait_timeout=self.backbone_timeout)
        self._to_worker_fli = self._attach_to_worker_queue()

        channels = self._create_worker_channels()
        self._from_worker_ch = channels[0]
        self._to_worker_ch = channels[1]

        self._publisher = self._create_broadcaster()

        self.perf_timer: PerfTimer = PerfTimer(
            debug=False, timing_on=timing_on, prefix=f"a{rank}_"
        )
        self._start: t.Optional[float] = None
        self._interm: t.Optional[float] = None
        self._timings: _TimingDict = OrderedDict()
        self._timing_on = timing_on

    @property
    def backbone_timeout(self) -> float:
        """The timeout (in seconds) applied to retrievals from the backbone feature store.

        :returns: A float indicating the number of seconds to allow"""
        return self._backbone_timeout or self._DEFAULT_BACKBONE_TIMEOUT

    def _add_label_to_timings(self, label: str) -> None:
        if label not in self._timings:
            self._timings[label] = []

    @staticmethod
    def _format_number(number: t.Union[numbers.Number, float]) -> str:
        return f"{number:0.4e}"

    def start_timings(self, batch_size: numbers.Number) -> None:
        if self._timing_on:
            self._add_label_to_timings("batch_size")
            self._timings["batch_size"].append(self._format_number(batch_size))
            self._start = time.perf_counter()
            self._interm = time.perf_counter()

    def end_timings(self) -> None:
        if self._timing_on and self._start is not None:
            self._add_label_to_timings("total_time")
            self._timings["total_time"].append(
                self._format_number(time.perf_counter() - self._start)
            )

    def measure_time(self, label: str) -> None:
        if self._timing_on and self._interm is not None:
            self._add_label_to_timings(label)
            self._timings[label].append(
                self._format_number(time.perf_counter() - self._interm)
            )
            self._interm = time.perf_counter()

    def print_timings(self, to_file: bool = False) -> None:
        print(" ".join(self._timings.keys()))

        value_array = numpy.array(self._timings.values(), dtype=float)
        value_array = numpy.transpose(value_array)
        for i in range(value_array.shape[0]):
            print(" ".join(self._format_number(value) for value in value_array[i]))
        if to_file:
            numpy.save("timings.npy", value_array)
            numpy.savetxt("timings.txt", value_array)

    def run_model(self, model: t.Union[bytes, str], batch: torch.Tensor) -> t.Any:
        tensors = [batch.numpy()]
        self.perf_timer.start_timings("batch_size", batch.shape[0])
        built_tensor_desc = MessageHandler.build_tensor_descriptor(
            "c", "float32", list(batch.shape)
        )
        self.perf_timer.measure_time("build_tensor_descriptor")
        if isinstance(model, str):
            model_arg = MessageHandler.build_model_key(model, self._backbone.descriptor)
        else:
            model_arg = MessageHandler.build_model(
                model, "resnet-50", "1.0"
            )  # type: ignore
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

        if self._to_worker_fli is None:
            raise ValueError("No worker queue available.")

        # pylint: disable-next=protected-access
        with self._to_worker_fli._channel.sendh(  # type: ignore
            timeout=None,
            stream_channel=self._to_worker_ch.channel,
        ) as to_sendh:
            to_sendh.send_bytes(request_bytes)
            self.perf_timer.measure_time("send_request")
            for tensor in tensors:
                to_sendh.send_bytes(tensor.tobytes())  # TODO NOT FAST ENOUGH!!!
                # to_sendh.send_bytes(bytes(tensor.data))
        logger.info(f"Message size: {len(request_bytes)} bytes")

        self.perf_timer.measure_time("send_tensors")
        with self._from_worker_ch.channel.recvh(timeout=None) as from_recvh:
            resp = from_recvh.recv_bytes(timeout=None)
            self.perf_timer.measure_time("receive_response")
            response = MessageHandler.deserialize_response(resp)
            self.perf_timer.measure_time("deserialize_response")
            # list of data blobs?
            # recv depending on the len(response.result.descriptors)?
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

    def set_model(self, key: str, model: bytes) -> None:
        # todo: incorrect usage of backbone here to store
        # user models? are we using the backbone if they do NOT
        # have a feature store of their own?
        self._backbone[key] = model

        # notify components of a change in the data at this key
        event = OnWriteFeatureStore(self._backbone.descriptor, key)
        self._publisher.send(event)
