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

import multiprocessing as mp
import numbers
import time
import typing as t
from collections import OrderedDict

import dragon
import numpy as np

from ....utils.timings import PerfTimer
from .....log import get_logger
from ....entrypoints.service import Service
from ...comm.channel.channel import CommChannelBase
from ...comm.channel.dragonchannel import DragonCommChannel
from ...infrastructure.environmentloader import EnvironmentConfigLoader
from ...infrastructure.storage.featurestore import FeatureStore
from ...infrastructure.worker.worker import (
    InferenceReply,
    InferenceRequest,
    LoadModelResult,
    MachineLearningWorkerBase,
)
from ...message_handler import MessageHandler
from ...mli_schemas.response.response_capnp import Response
from .devicemanager import DeviceManager, WorkerDevice
from .requestdispatcher import RequestDispatcher

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.model.model_capnp import Model
    from smartsim._core.mli.mli_schemas.response.response_capnp import StatusEnum

logger = get_logger(__name__)


def build_failure_reply(status: "StatusEnum", message: str) -> Response:
    return MessageHandler.build_response(
        status=status,  # todo: need to indicate correct status
        message=message,  # todo: decide what these will be
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
        arrays: t.List[np.ndarray[t.Any, np.dtype[t.Any]]] = [
            output.numpy() for output in reply.outputs
        ]
        for tensor in arrays:
            # todo: need to have the output attributes specified in the req?
            # maybe, add `MessageHandler.dtype_of(tensor)`?
            # can `build_tensor` do dtype and shape?
            msg_tensor = MessageHandler.build_tensor(
                tensor,
                "c",
                "float32",
                [1],
            )
            prepared_outputs.append(msg_tensor)
    return prepared_outputs


def build_reply(reply: InferenceReply) -> Response:
    results = prepare_outputs(reply)

    return MessageHandler.build_response(
        status="complete",
        message="success",
        result=results,
        custom_attributes=None,
    )


class WorkerManager(Service):
    """An implementation of a service managing distribution of tasks to
    machine learning workers"""

    def __init__(
        self,
        config_loader: EnvironmentConfigLoader,
        worker: MachineLearningWorkerBase,
        as_service: bool = False,
        cooldown: int = 0,
        comm_channel_type: t.Type[CommChannelBase] = DragonCommChannel,
        device: t.Literal["cpu", "gpu"] = "cpu",
        batch_timeout: float = 0.0,
        batch_size: int = 1,
    ) -> None:
        """Initialize the WorkerManager
        :param config_loader: Environment config loader that loads the task queue and
        feature store
        :param workers: A worker to manage
        :param as_service: Specifies run-once or run-until-complete behavior of service
        :param cooldown: Number of seconds to wait before shutting down after
        shutdown criteria are met
        :param comm_channel_type: The type of communication channel used for callbacks
        """
        super().__init__(as_service, cooldown)

        self._task_queue: t.Optional[CommChannelBase] = config_loader.get_queue()
        """the queue the manager monitors for new tasks"""
        self._feature_store: t.Optional[FeatureStore] = (
            config_loader.get_feature_store()
        )
        """a feature store to retrieve models from"""
        self._worker = worker
        """The ML Worker implementation"""
        self._comm_channel_type = comm_channel_type
        """The type of communication channel to construct for callbacks"""
        self._device = device
        """Device on which workers need to run"""
        self._cached_models: dict[str, t.Any] = {}
        """Dictionary of previously loaded models"""
        self._request_dispatcher: RequestDispatcher = RequestDispatcher(
            batch_timeout=batch_timeout,
            batch_size=batch_size,
            incoming_channel=self._task_queue,
            comm_channel_type=comm_channel_type,
            feature_store=self._feature_store,
        )
        """Dispatcher used to batch requests"""
        self._device_manager: DeviceManager = DeviceManager([WorkerDevice("gpu")])

        self._perf_timer = PerfTimer(prefix="w_")

        try:
            mp.set_start_method("dragon")
        except RuntimeError:
            pass
        self._dispatcher_process = mp.Process(
            target=self._request_dispatcher.run, name="Dispatcher"
        )

    def _on_start(self) -> None:
        self._dispatcher_process.start()

    def _on_shutdown(self) -> None:
        self._dispatcher_process.join()

    def _validate_request(self, request: InferenceRequest) -> bool:
        """Ensure the request can be processed.
        :param request: The request to validate
        :return: True if the request is valid, False otherwise"""
        if not self._feature_store:
            if request.model_key:
                logger.error("Unable to load model by key without feature store")
                return False

            if request.input_keys:
                logger.error("Unable to load inputs by key without feature store")
                return False

            if request.output_keys:
                logger.error("Unable to persist outputs by key without feature store")
                return False

        if not request.model_key and not request.raw_model:
            logger.error("Unable to continue without model bytes or feature store key")
            return False

        if not request.input_keys and not request.raw_inputs:
            logger.error("Unable to continue without input bytes or feature store keys")
            return False

        if request.callback is None:
            logger.error("No callback channel provided in request")
            return False

        return True

    def _on_iteration(self) -> None:
        """Executes calls to the machine learning worker implementation to complete
        the inference pipeline"""
        logger.debug("executing worker manager pipeline")

        batch = self._request_dispatcher.task_queue.get()
        self._perf_timer.start_timings()
        if batch is None or 0 == len(batch.requests):
            return

        self._perf_timer.measure_time("flush_requests")
        # logger.info(f"Got batch of {len(batch.requests)} requests, acquiring device")
        device: WorkerDevice = next(
            self._device_manager.get_free_device(
                worker=self._worker,
                batch=batch,
                feature_store=self._feature_store,
            )
        )
        self._perf_timer.measure_time("fetch_model")

        # logger.info(f"Acquired device {device.name}")

        model_result = LoadModelResult(device.get_model(batch.model_key))
        self._perf_timer.measure_time("load_model")

        fetch_input_results = self._worker.fetch_inputs(batch, self._feature_store)
        self._perf_timer.measure_time("fetch_input")

        transformed_input = self._worker.transform_input(
            batch, fetch_input_results, self._device
        )
        self._perf_timer.measure_time("transform_input")

        replies = [InferenceReply() for _ in range(len(batch.requests))]

        try:
            execute_result = self._worker.execute(
                batch, model_result, transformed_input
            )
            self._perf_timer.measure_time("execute")
            transformed_outputs = self._worker.transform_output(
                batch, execute_result, self._device
            )
            self._perf_timer.measure_time("transform_output")
        except Exception:
            logger.exception("Error executing worker")
            for reply in replies:
                reply.failed = True
        else:
            for reply_idx, (request, transformed_output) in enumerate(
                zip(batch.requests, transformed_outputs)
            ):
                reply = replies[reply_idx]
                try:
                    if request.output_keys:
                        reply.output_keys = self._worker.place_output(
                            request, transformed_output, self._feature_store
                        )
                    else:
                        reply.outputs = transformed_output.outputs
                    self._perf_timer.measure_time("assign_output")
                except Exception:
                    logger.exception("Error executing worker")
                    reply.failed = True

                if reply.failed:
                    response = build_failure_reply("fail", "failure-occurred")
                else:
                    if reply.outputs is None or not reply.outputs:
                        response = build_failure_reply("fail", "no-results")

                response = build_reply(reply)
                self._perf_timer.measure_time("build_reply")

                serialized_resp = MessageHandler.serialize_response(response)  # type: ignore

                self._perf_timer.measure_time("serialize_resp")

                if request.callback:
                    request.callback.send(serialized_resp)
                self._perf_timer.measure_time("send")

        self._perf_timer.end_timings()

        if len(self._perf_timer._timings["w_send"]) == 801:
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
