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
import dragon.channels
from dragon.globalservices.api_setup import connect_to_infrastructure

try:
    from mpi4py import MPI  # type: ignore[import-not-found]
except Exception:
    MPI = None
    print("Unable to import `mpi4py` package")

# isort: on
# pylint: enable=unused-import,import-error

import numbers
import os
import time
import typing as t
from collections import OrderedDict

import numpy
import torch

from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
from smartsim._core.mli.comm.channel.dragon_util import create_local
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
    def _attach_to_backbone() -> BackboneFeatureStore:
        """Use the supplied environment variables to attach
        to a pre-existing backbone featurestore. Requires the
        environment to contain `_SMARTSIM_INFRA_BACKBONE`
        environment variable.

        :returns: The attached backbone featurestore
        """
        # todo: ensure this env var from config loader or constant
        descriptor = os.environ.get(BackboneFeatureStore.MLI_BACKBONE, None)
        if descriptor is None or not descriptor:
            raise SmartSimError(
                "Missing required backbone configuration in environment: "
                f"{BackboneFeatureStore.MLI_BACKBONE}"
            )

        backbone = t.cast(
            BackboneFeatureStore, BackboneFeatureStore.from_descriptor(descriptor)
        )
        return backbone

    def _attach_to_worker_queue(self) -> DragonFLIChannel:
        """Wait until the backbone contains the worker queue configuration,
        then attach an FLI to the given worker queue.

        :returns: The attached FLI channel
        :raises: SmartSimError if the required configuration is not found in the
        backbone feature store
        """

        descriptor = ""
        try:
            # NOTE: without wait_for, this MUST be in the backbone....
            config = self._backbone.wait_for(
                [BackboneFeatureStore.MLI_WORKER_QUEUE], self.backbone_timeout
            )
            descriptor = str(config[BackboneFeatureStore.MLI_WORKER_QUEUE])
        except Exception as ex:
            logger.info(
                f"Unable to retrieve {BackboneFeatureStore.MLI_WORKER_QUEUE} "
                "to attach to the worker queue."
            )
            raise SmartSimError("Unable to locate worker queue using backbone") from ex

        return DragonFLIChannel.from_descriptor(descriptor)

    def _create_broadcaster(self) -> EventProducer:
        """Create an event publisher that will broadcast updates to
        other MLI components. This publisher

        :returns: the event publisher instance
        """
        broadcaster = EventBroadcaster(
            self._backbone, DragonCommChannel.from_descriptor
        )
        return broadcaster

    def __init__(self, timing_on: bool, wait_timeout: float = 0) -> None:
        """Initialize the client instance.

        :param timing_on: Flag indicating if timing information should be
        written to file
        :param wait_timeout: Maximum wait time (in seconds) allowed to attach to the
        worker queue
        :raises: SmartSimError if unable to attach to a backbone featurestore
        """
        if MPI is not None:
            # todo: determine a way to make MPI work in the test environment
            #  - consider catching the import exception and defaulting rank to 0
            comm = MPI.COMM_WORLD
            rank: int = comm.Get_rank()
        else:
            rank = 0

        self._backbone_timeout = wait_timeout

        connect_to_infrastructure()

        self._backbone = self._attach_to_backbone()
        self._backbone.wait_timeout = self.backbone_timeout
        self._to_worker_fli = self._attach_to_worker_queue()

        self._from_worker_ch = create_local(self._DEFAULT_WORK_QUEUE_SIZE)
        self._to_worker_ch = create_local(self._DEFAULT_WORK_QUEUE_SIZE)

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
        """The timeout (in seconds) applied to retrievals
        from the backbone feature store.

        :returns: A float indicating the number of seconds to allow"""
        return self._backbone_timeout or self._DEFAULT_BACKBONE_TIMEOUT

    def _add_label_to_timings(self, label: str) -> None:
        """Adds a new label into the timing dictionary to prepare for
        receiving timing events.

        :param label: The label to create storage for
        """
        if label not in self._timings:
            self._timings[label] = []

    @staticmethod
    def _format_number(number: t.Union[numbers.Number, float]) -> str:
        """Utility function for formatting numbers consistently for logs.

        :param number: The number to convert to a formatted string
        :returns: The formatted string containing the number
        """
        return f"{number:0.4e}"

    def start_timings(self, batch_size: numbers.Number) -> None:
        """Configure the client to begin storing timing information.

        :param bach_size: The size of batches to generate as inputs
        to the model
        """
        if self._timing_on:
            self._add_label_to_timings("batch_size")
            self._timings["batch_size"].append(self._format_number(batch_size))
            self._start = time.perf_counter()
            self._interm = time.perf_counter()

    def end_timings(self) -> None:
        """Configure the client to stop storing timing information."""
        if self._timing_on and self._start is not None:
            self._add_label_to_timings("total_time")
            self._timings["total_time"].append(
                self._format_number(time.perf_counter() - self._start)
            )

    def measure_time(self, label: str) -> None:
        """Measures elapsed time since the last recorded signal.

        :param label: The label to measure time for
        """
        if self._timing_on and self._interm is not None:
            self._add_label_to_timings(label)
            self._timings[label].append(
                self._format_number(time.perf_counter() - self._interm)
            )
            self._interm = time.perf_counter()

    def print_timings(self, to_file: bool = False) -> None:
        """Print timing information to standard output.

        :param to_file: If `True`, also saves timing information
        to the files `timings.npy` and `timings.txt`
        """
        print(" ".join(self._timings.keys()))

        value_array = numpy.array(self._timings.values(), dtype=float)
        value_array = numpy.transpose(value_array)
        for i in range(value_array.shape[0]):
            print(" ".join(self._format_number(value) for value in value_array[i]))
        if to_file:
            numpy.save("timings.npy", value_array)
            numpy.savetxt("timings.txt", value_array)

    def run_model(self, model: t.Union[bytes, str], batch: torch.Tensor) -> t.Any:
        """Execute a bach of inference requests with the supplied ML model.

        :param model: The raw bytes or path to a pytorch model
        :param batch: The tensor batch to perform inference on
        :returns: The inference results
        :raises: ValueError if the worker queue is not configured properly
        in the environment variables
        """
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
        """Write the supplied model to the feature store.

        :param key: The unique key used to identify the model
        :param model: The raw bytes of the model to execute
        """
        self._backbone[key] = model

        # notify components of a change in the data at this key
        event = OnWriteFeatureStore(self._backbone.descriptor, key)
        self._publisher.send(event)
