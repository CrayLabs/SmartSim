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
import dragon
from dragon import fli

# isort: on

import time
import typing as t

import numpy as np

from .....error import SmartSimError
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

if t.TYPE_CHECKING:
    from dragon.fli import FLInterface

    from smartsim._core.mli.mli_schemas.model.model_capnp import Model
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status

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
    model_bytes: t.Optional[Model] = None

    if request.model.which() == "key":
        model_key = request.model.key.key
    elif request.model.which() == "data":
        model_bytes = request.model.data

    callback_key = request.replyChannel.reply

    # todo: shouldn't this be `CommChannel.find` instead of `DragonCommChannel`
    comm_channel = channel_type(callback_key)
    # comm_channel = DragonCommChannel(request.replyChannel)

    input_keys: t.Optional[t.List[str]] = None
    input_bytes: t.Optional[t.List[bytes]] = (
        None  # these will really be tensors already
    )
    output_keys: t.Optional[t.List[str]] = None

    input_meta: t.List[t.Any] = []

    if request.input.which() == "keys":
        input_keys = [input_key.key for input_key in request.input.keys]
    elif request.input.which() == "data":
        input_bytes = [data.blob for data in request.input.data]
        input_meta = [data.tensorDescriptor for data in request.input.data]

    if request.output:
        output_keys = [tensor_key.key for tensor_key in request.output]

    inference_request = InferenceRequest(
        model_key=model_key,
        callback=comm_channel,
        raw_inputs=input_bytes,
        input_keys=input_keys,
        input_meta=input_meta,
        output_keys=output_keys,
        raw_model=model_bytes,
        batch_size=0,
    )
    return inference_request


def build_failure_reply(status: "Status", message: str) -> Response:
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
        status=reply.status_enum,
        message=reply.message,
        result=results,
        custom_attributes=None,
    )


def exception_handler(
    exc: Exception, reply_channel: t.Optional[CommChannelBase], failure_message: str
) -> None:
    """
    Logs exceptions, calls send_failure to send the failed response back.

    :param exc: The exception to be logged
    :param reply_channel: The channel used to send replies
    :param failure_message: Failure message to log and send back
    """
    logger.exception(
        f"{failure_message}\n"
        f"Exception type: {type(exc).__name__}\n"
        f"Exception message: {str(exc)}"
    )
    send_failure(reply_channel, failure_message)


def send_failure(
    reply_channel: t.Optional[CommChannelBase], failure_message: str
) -> None:
    """
    Sends back the failed response.

    :param reply_channel: The channel used to send replies
    :param failure_message: Failure message for response
    """
    serialized_resp = MessageHandler.serialize_response(build_failure_reply("fail", failure_message))  # type: ignore
    if reply_channel:
        reply_channel.send(serialized_resp)


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

        timings = []  # timing
        # perform default deserialization of the message envelope
        request_bytes: bytes = self._task_queue.recv()

        interm = time.perf_counter()  # timing
        request = deserialize_message(
            request_bytes, self._comm_channel_type, self._device
        )
        if not self._validate_request(request):
            return

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing

        reply = InferenceReply()

        if not request.raw_model:
            if request.model_key is None:
                send_failure(request.callback, "Could not find model key or model.")
                return
            if request.model_key in self._cached_models:
                timings.append(time.perf_counter() - interm)  # timing
                interm = time.perf_counter()  # timing
                model_result = LoadModelResult(self._cached_models[request.model_key])

            else:
                fetch_model_result = None
                while fetch_model_result is None:
                    try:
                        interm = time.perf_counter()  # timing
                        fetch_model_result = self._worker.fetch_model(
                            request, self._feature_store
                        )

                    # do we want to keep this? it could cause an infinite loop
                    except KeyError:
                        time.sleep(0.01)
                    except Exception as e:
                        exception_handler(
                            e, request.callback, "Failed while fetching the model."
                        )
                        return

                timings.append(time.perf_counter() - interm)  # timing
                interm = time.perf_counter()  # timing
                try:
                    model_result = self._worker.load_model(
                        request,
                        fetch_result=fetch_model_result,
                        device=self._device,
                    )
                    self._cached_models[request.model_key] = model_result.model
                except Exception as e:
                    exception_handler(
                        e, request.callback, "Failed while loading the model."
                    )
                    return

        else:
            try:
                fetch_model_result = self._worker.fetch_model(
                    request, self._feature_store
                )
            except Exception as e:
                exception_handler(
                    e, request.callback, "Failed while fetching the model."
                )
                return
            try:
                model_result = self._worker.load_model(
                    request, fetch_result=fetch_model_result, device=self._device
                )
            except Exception as e:
                exception_handler(
                    e, request.callback, "Failed while loading the model."
                )
                return

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing
        try:
            fetch_input_result = self._worker.fetch_inputs(request, self._feature_store)
        except Exception as e:
            exception_handler(e, request.callback, "Failed while fetching the inputs.")
            return

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing
        try:
            transformed_input = self._worker.transform_input(
                request, fetch_input_result, self._device
            )
        except Exception as e:
            exception_handler(
                e, request.callback, "Failed while transforming the input."
            )
            return

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing

        try:
            execute_result = self._worker.execute(
                request, model_result, transformed_input
            )
        except Exception as e:
            exception_handler(e, request.callback, "Failed while executing.")
            return

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing
        try:
            transformed_output = self._worker.transform_output(
                request, execute_result, self._device
            )
        except Exception as e:
            exception_handler(
                e, request.callback, "Failed while transforming the output."
            )
            return

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing
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
                return
        else:
            reply.outputs = transformed_output.outputs

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing

        if reply.outputs is None or not reply.outputs:
            response = build_failure_reply("fail", "Outputs not found.")
        else:
            reply.status_enum = "complete"
            reply.message = "Success"
            response = build_reply(reply)

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing

        serialized_resp = MessageHandler.serialize_response(response)  # type: ignore

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing
        if request.callback:
            request.callback.send(serialized_resp)

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing

        print(" ".join(str(time) for time in timings))  # timing

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
