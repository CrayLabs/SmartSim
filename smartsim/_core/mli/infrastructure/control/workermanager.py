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

# pylint: disable=import-error
# pylint: disable-next=unused-import
import dragon

# pylint: enable=import-error

# isort: off
# isort: on

import multiprocessing as mp
import time
import typing as t

from .....log import get_logger
from ....entrypoints.service import Service
from ....utils.timings import PerfTimer
from ...comm.channel.channel import CommChannelBase
from ...comm.channel.dragonchannel import DragonCommChannel
from ...infrastructure.environmentloader import EnvironmentConfigLoader
from ...infrastructure.storage.featurestore import FeatureStore
from ...infrastructure.worker.worker import (
    InferenceBatch,
    InferenceReply,
    LoadModelResult,
    MachineLearningWorkerBase,
)
from ...message_handler import MessageHandler
from ...mli_schemas.response.response_capnp import ResponseBuilder
from .devicemanager import DeviceManager, WorkerDevice

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.model.model_capnp import Model
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status
    from smartsim._core.mli.mli_schemas.tensor.tensor_capnp import TensorDescriptor

logger = get_logger(__name__)


def build_failure_reply(status: "Status", message: str) -> ResponseBuilder:
    return MessageHandler.build_response(
        status=status,
        message=message,
        result=[],
        custom_attributes=None,
    )


def prepare_outputs(reply: InferenceReply) -> t.List[t.Any]:
    prepared_outputs: t.List[t.Any] = []
    if reply.output_keys:
        for key in reply.output_keys:
            if not key:
                continue
            msg_key = MessageHandler.build_tensor_key(key)
            prepared_outputs.append(msg_key)
    elif reply.outputs:
        for _ in reply.outputs:
            msg_tensor_desc = MessageHandler.build_tensor_descriptor(
                "c",
                "float32",
                [1],
            )
            prepared_outputs.append(msg_tensor_desc)
    return prepared_outputs


def build_reply(reply: InferenceReply) -> ResponseBuilder:
    results = prepare_outputs(reply)

    return MessageHandler.build_response(
        status=reply.status_enum,
        message=reply.message,
        result=results,
        custom_attributes=None,
    )


def exception_handler(
    exc: Exception, reply_channel: t.Optional[CommChannelBase], failure_message: str
) -> None:
    """
    Logs exceptions and sends a failure response.

    :param exc: The exception to be logged
    :param reply_channel: The channel used to send replies
    :param failure_message: Failure message to log and send back
    """
    logger.exception(
        f"{failure_message}\n"
        f"Exception type: {type(exc).__name__}\n"
        f"Exception message: {str(exc)}"
    )
    serialized_resp = MessageHandler.serialize_response(
        build_failure_reply("fail", failure_message)
    )
    if reply_channel:
        reply_channel.send(serialized_resp)


class WorkerManager(Service):
    """An implementation of a service managing distribution of tasks to
    machine learning workers"""

    def __init__(
        self,
        config_loader: EnvironmentConfigLoader,
        worker_type: t.Type[MachineLearningWorkerBase],
        task_queue: "mp.Queue[InferenceBatch]",
        as_service: bool = False,
        cooldown: int = 0,
        comm_channel_type: t.Type[CommChannelBase] = DragonCommChannel,
        device: t.Literal["cpu", "gpu"] = "cpu",
    ) -> None:
        """Initialize the WorkerManager
        :param config_loader: Environment config loader that loads the task queue and
        feature store
        :param worker_type: The type of worker to manage
        :param task_queue: Queue from witch the batched requests have to be pulled
        :param as_service: Specifies run-once or run-until-complete behavior of service
        :param cooldown: Number of seconds to wait before shutting down after
        shutdown criteria are met
        :param comm_channel_type: The type of communication channel used for callbacks
        :param device: The device on which the Worker should run. Every worker manager
        is assigned one single GPU (if available), thus the device should have no index.
        """
        super().__init__(as_service, cooldown)

        self._task_queue = task_queue
        """The dispatcher queue the manager monitors for new tasks"""
        self._feature_store: t.Optional[FeatureStore] = (
            config_loader.get_feature_store()
        )
        """A feature store to retrieve models from"""
        self._worker = worker_type()
        """The ML Worker implementation"""
        self._comm_channel_type = comm_channel_type
        """The type of communication channel to construct for callbacks"""
        self._device = device
        """Device on which workers need to run"""

        self._device_manager: t.Optional[DeviceManager] = None
        """Object responsible for model caching and device access"""
        self._perf_timer = PerfTimer(prefix="w_", debug=False, timing_on=True)
        """Performance timer"""

    def _on_start(self) -> None:
        self._device_manager = DeviceManager([WorkerDevice(self._device)])

    # pylint: disable-next=too-many-statements
    def _on_iteration(self) -> None:
        """Executes calls to the machine learning worker implementation to complete
        the inference pipeline"""

        pre_batch_time = time.perf_counter()
        try:
            batch: InferenceBatch = self._task_queue.get(timeout=0.0001)
        except Exception:
            return

        self._perf_timer.start_timings(
            "flush_requests", time.perf_counter() - pre_batch_time
        )

        if batch is None or 0 == len(batch.requests):
            return

        if self._device_manager is None:
            raise ValueError("No Device Manager available: did you call _on_start()")
        device: WorkerDevice = next(
            self._device_manager.get_free_device(
                worker=self._worker,
                batch=batch,
                feature_store=self._feature_store,
            )
        )
        self._perf_timer.measure_time("fetch_model")

        model_result = LoadModelResult(device.get_model(batch.model_key))
        self._perf_timer.measure_time("load_model")

        if batch.inputs is None:
            for request in batch.requests:
                exception_handler(
                    ValueError("Error batching inputs"),
                    request.callback,
                    "Error batching inputs.",
                )
            return
        transformed_input = batch.inputs

        try:
            execute_result = self._worker.execute(
                batch, model_result, transformed_input, device.name
            )
        except Exception as e:
            for request in batch.requests:
                exception_handler(e, request.callback, "Error executing worker.")
            return
        self._perf_timer.measure_time("execute")

        try:
            transformed_outputs = self._worker.transform_output(batch, execute_result)
        except Exception as e:
            for request in batch.requests:
                exception_handler(
                    e, request.callback, "Failed while transforming the output."
                )
            return

        for request, transformed_output in zip(batch.requests, transformed_outputs):
            reply = InferenceReply()
            if request.output_keys:
                try:
                    reply.output_keys = self._worker.place_output(
                        request,
                        transformed_output,
                        self._feature_store,
                    )
                except Exception as e:
                    exception_handler(
                        e, request.callback, "Failed while placing the output."
                    )
                    continue
            else:
                reply.outputs = transformed_output.outputs
            self._perf_timer.measure_time("assign_output")

            if reply.outputs is None:
                response = build_failure_reply("fail", "Outputs not found.")
            else:
                reply.status_enum = "complete"
                reply.message = "Success"
                response = build_reply(reply)

            self._perf_timer.measure_time("build_reply")

            serialized_resp = MessageHandler.serialize_response(response)

            self._perf_timer.measure_time("serialize_resp")

            if request.callback:
                request.callback.send(serialized_resp)
                if reply.outputs:
                    # send tensor data after response
                    for output in reply.outputs:
                        request.callback.send(output)
            self._perf_timer.measure_time("send")

        self._perf_timer.end_timings()

        if self._perf_timer.max_length == 801:
            self._perf_timer.print_timings(True)

    def _can_shutdown(self) -> bool:
        """Return true when the criteria to shut down the service are met."""
        # todo: determine shutdown criteria
        # will we receive a completion message?
        # will we let MLI mgr just kill this?
        # time_diff = self._last_event - datetime.datetime.now()
        # if time_diff.total_seconds() > self._cooldown:
        #     return True
        # return False
        return self._worker is None
