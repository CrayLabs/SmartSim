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

import sys

# isort: off
try:
    import dragon
    from dragon import fli
except ImportError as exc:
    if not "pytest" in sys.modules:
        raise exc from None

# isort: on
import time
import typing as t

import numpy as np

from smartsim._core.entrypoints.service import Service
from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim._core.mli.comm.channel.dragonfli import DragonFLIChannel
from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore
from smartsim._core.mli.infrastructure.worker.worker import (
    InferenceReply,
    InferenceRequest,
    MachineLearningWorkerBase,
)
from smartsim._core.mli.message_handler import MessageHandler
from smartsim._core.mli.mli_schemas.response.response_capnp import Response
from smartsim.log import get_logger

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.response.response_capnp import StatusEnum

logger = get_logger(__name__)


def deserialize_message(
    data_blob: bytes,
    channel_type: t.Type[CommChannelBase],
    device: t.Literal["cpu", "gpu"],
) -> InferenceRequest:
    """Deserialize a message from a byte stream into an InferenceRequest
    :param data_blob: The byte stream to deserialize"""
    # todo: consider moving to XxxCore and only making
    # workers implement the inputs and model conversion?

    # alternatively, consider passing the capnproto models
    # to this method instead of the data_blob...

    # something is definitely wrong here... client shouldn't have to touch
    # callback (or batch size)

    request = MessageHandler.deserialize_request(data_blob)
    # return request
    model_key: t.Optional[str] = None
    model_bytes: t.Optional[bytes] = None

    if request.model.which() == "modelKey":
        model_key = request.model.modelKey.key
    elif request.model.which() == "modelData":
        model_bytes = request.model.modelData

    callback_key = request.replyChannel.reply

    # todo: shouldn't this be `CommChannel.find` instead of `DragonCommChannel`
    comm_channel = channel_type(callback_key)
    # comm_channel = DragonCommChannel(request.replyChannel)

    input_keys: t.Optional[t.List[str]] = None
    input_bytes: t.Optional[t.List[bytes]] = (
        None  # these will really be tensors already
    )

    input_meta: t.List[t.Any] = []

    if request.input.which() == "inputKeys":
        input_keys = [input_key.key for input_key in request.input.inputKeys]
    elif request.input.which() == "inputData":
        input_bytes = [data.blob for data in request.input.inputData]
        input_meta = [data.tensorDescriptor for data in request.input.inputData]

    inference_request = InferenceRequest(
        model_key=model_key,
        callback=comm_channel,
        raw_inputs=input_bytes,
        input_meta=input_meta,
        input_keys=input_keys,
        raw_model=model_bytes,
        batch_size=0,
    )
    return inference_request


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
        file_like_interface: fli.FLInterface,
        worker: MachineLearningWorkerBase,
        feature_store: t.Optional[FeatureStore] = None,
        as_service: bool = False,
        cooldown: int = 0,
        comm_channel_type: t.Type[CommChannelBase] = DragonFLIChannel,
        device: t.Literal["cpu", "gpu"] = "cpu",
    ) -> None:
        """Initialize the WorkerManager
        :param task_queue: The queue to monitor for new tasks
        :param workers: A worker to manage
        :param feature_store: The persistence mechanism
        :param as_service: Specifies run-once or run-until-complete behavior of service
        :param cooldown: Number of seconds to wait before shutting down afer
        shutdown criteria are met
        :param comm_channel_type: The type of communication channel used for callbacks
        """
        super().__init__(as_service, cooldown)

        """a collection of workers the manager is controlling"""
        self._task_queue: fli.FLInterface = file_like_interface
        """the queue the manager monitors for new tasks"""
        self._feature_store: t.Optional[FeatureStore] = feature_store
        """a feature store to retrieve models from"""
        self._worker = worker
        """The ML Worker implementation"""
        self._comm_channel_type = comm_channel_type
        """The type of communication channel to construct for callbacks"""
        self._device = device
        """Device on which workers need to run"""

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

        if self._task_queue is None:
            logger.warning("No queue to check for tasks")
            return

        # perform default deserialization of the message envelope
        # perform default deserialization of the message envelope
        with self._task_queue.recvh(timeout=None) as recvh:
            try:
                request_bytes, _ = recvh.recv_bytes(timeout=None)
            except fli.FLIEOT as exc:
                return

        request = deserialize_message(
            request_bytes, self._comm_channel_type, self._device
        )
        if not self._validate_request(request):
            return

        # # let the worker perform additional custom deserialization
        # request = self._worker.deserialize(request_bytes)

        fetch_model_result = self._worker.fetch_model(request, self._feature_store)
        model_result = self._worker.load_model(
            request, fetch_model_result, self._device
        )
        fetch_input_result = self._worker.fetch_inputs(request, self._feature_store)
        transformed_input = self._worker.transform_input(
            request, fetch_input_result, self._device
        )

        reply = InferenceReply()

        try:
            execute_result = self._worker.execute(
                request, model_result, transformed_input
            )
            transformed_output = self._worker.transform_output(
                request, execute_result, self._device
            )

            if request.output_keys:
                reply.output_keys = self._worker.place_output(
                    request, transformed_output, self._feature_store
                )
            else:
                reply.outputs = transformed_output.outputs
        except Exception:
            logger.exception("Error executing worker")
            reply.failed = True

        if reply.failed:
            response = build_failure_reply("fail", "failure-occurred")
        else:
            if reply.outputs is None or not reply.outputs:
                response = build_failure_reply("fail", "no-results")

            response = build_reply(reply)

        # serialized = self._worker.serialize_reply(request, transformed_output)
        serialized_resp = MessageHandler.serialize_response(response)  # type: ignore
        if request.callback:
            request.callback.send(serialized_resp)

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
