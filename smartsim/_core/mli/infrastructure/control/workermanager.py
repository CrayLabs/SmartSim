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

import time
import typing as t

from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore

from .....log import get_logger
from ....entrypoints.service import Service
from ...comm.channel.channel import CommChannelBase
from ...comm.channel.dragonchannel import DragonCommChannel
from ...infrastructure.environmentloader import EnvironmentConfigLoader
from ...infrastructure.worker.worker import (
    InferenceReply,
    InferenceRequest,
    LoadModelResult,
    MachineLearningWorkerBase,
)
from ...message_handler import MessageHandler
from ...mli_schemas.response.response_capnp import ResponseBuilder

if t.TYPE_CHECKING:
    from dragon.fli import FLInterface
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status

logger = get_logger(__name__)


def build_failure_reply(status: "Status", message: str) -> ResponseBuilder:
    """Build a response indicating a failure occurred
    :param status: The status of the response
    :param message: The error message to include in the response"""
    return MessageHandler.build_response(
        status=status,
        message=message,
        result=None,
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
        worker: MachineLearningWorkerBase,
        as_service: bool = False,
        cooldown: int = 0,
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
        self._worker = worker
        """The ML Worker implementation"""
        self._callback_factory = config_loader._callback_factory
        """The type of communication channel to construct for callbacks"""
        self._device = device
        """Device on which workers need to run"""
        self._cached_models: dict[str, t.Any] = {}
        """Dictionary of previously loaded models"""
        self._feature_stores: t.Dict[str, FeatureStore] = {}
        """A collection of attached feature stores"""
        self._fs_factory = config_loader._featurestore_factory
        """A factory method to create a desired feature store client type"""
        self._backbone: t.Optional[FeatureStore] = config_loader.get_backbone()
        """The backbone feature store"""

    def _check_feature_stores(self, request: InferenceRequest) -> bool:
        """Ensures that all feature stores required by the request are available
        :param request: The request to validate"""
        # collect all feature stores required by the request
        fs_model: t.Set[str] = set()
        if request.model_key:
            fs_model = {request.model_key.descriptor}
        fs_inputs = {key.descriptor for key in request.input_keys}
        fs_outputs = {key.descriptor for key in request.output_keys}

        # identify which feature stores are requested and unknown
        fs_desired = fs_model.union(fs_inputs).union(fs_outputs)
        fs_actual = {item.descriptor for item in self._feature_stores.values()}
        fs_missing = fs_desired - fs_actual

        if self._fs_factory is None:
            logger.warning("No feature store factory configured")
            return False

        # create the feature stores we need to service request
        if fs_missing:
            logger.info(f"Missing feature store(s): {fs_missing}")
            for descriptor in fs_missing:
                feature_store = self._fs_factory(descriptor)
                self._feature_stores[descriptor] = feature_store

        return True

    def _check_model(self, request: InferenceRequest) -> bool:
        """Ensure that a model is available for the request
        :param request: The request to validate"""
        if request.model_key or request.raw_model:
            return True

        logger.error("Unable to continue without model bytes or feature store key")
        return False

    def _check_inputs(self, request: InferenceRequest) -> bool:
        """Ensure that inputs are available for the request
        :param request: The request to validate"""
        if request.input_keys or request.raw_inputs:
            return True

        logger.error("Unable to continue without input bytes or feature store keys")
        return False

    def _check_callback(self, request: InferenceRequest) -> bool:
        """Ensure that a callback channel is available for the request
        :param request: The request to validate"""
        if request.callback is not None:
            return True

        logger.error("No callback channel provided in request")
        return False

    def _validate_request(self, request: InferenceRequest) -> bool:
        """Ensure the request can be processed.
        :param request: The request to validate
        :return: True if the request is valid, False otherwise"""
        checks = [
            self._check_feature_stores(request),
            self._check_model(request),
            self._check_inputs(request),
            self._check_callback(request),
        ]

        return all(checks)

    def _on_iteration(self) -> None:
        """Executes calls to the machine learning worker implementation to complete
        the inference pipeline"""
        logger.debug("executing worker manager pipeline")

        if self._task_queue is None:
            logger.warning("No queue to check for tasks")
            return

        timings = []  # timing

        bytes_list: t.List[bytes] = self._task_queue.recv()

        if not bytes_list:
            exception_handler(
                ValueError("No request data found"),
                None,
                "No request data found.",
            )
            return

        request_bytes = bytes_list[0]
        tensor_bytes_list = bytes_list[1:]

        interm = time.perf_counter()  # timing
        request = self._worker.deserialize_message(
            request_bytes, self._callback_factory
        )

        if request.input_meta and tensor_bytes_list:
            request.raw_inputs = tensor_bytes_list

        if not self._validate_request(request):
            return

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing

        reply = InferenceReply()

        if not request.raw_model:
            if request.model_key is None:
                exception_handler(
                    ValueError("Could not find model key or model"),
                    request.callback,
                    "Could not find model key or model.",
                )
                return

            # if request.model_key.descriptor not in self._feature_stores:
            #     self._fs_factory(request.model_key.descriptor)
            # todo: decide if we should load here or in _check_feature_stores.
            # todo: should i raise error here?

            if request.model_key.key in self._cached_models:
                timings.append(time.perf_counter() - interm)  # timing
                interm = time.perf_counter()  # timing
                model_result = LoadModelResult(
                    self._cached_models[request.model_key.key]
                )

            else:
                timings.append(time.perf_counter() - interm)  # timing
                interm = time.perf_counter()  # timing
                try:
                    fetch_model_result = self._worker.fetch_model(
                        request, self._feature_stores
                    )
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
                    self._cached_models[request.model_key.key] = model_result.model
                except Exception as e:
                    exception_handler(
                        e,
                        request.callback,
                        "Failed while loading model from feature store.",
                    )
                    return

        else:
            timings.append(time.perf_counter() - interm)  # timing
            interm = time.perf_counter()  # timing
            try:
                fetch_model_result = self._worker.fetch_model(
                    request, self._feature_stores
                )
            except Exception as e:
                exception_handler(
                    e, request.callback, "Failed while fetching the model."
                )
                return

            timings.append(time.perf_counter() - interm)  # timing
            interm = time.perf_counter()  # timing
            try:
                model_result = self._worker.load_model(
                    request, fetch_result=fetch_model_result, device=self._device
                )
            except Exception as e:
                exception_handler(
                    e,
                    request.callback,
                    "Failed while loading model from feature store.",
                )
                return

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing
        try:
            fetch_input_result = self._worker.fetch_inputs(
                request, self._feature_stores
            )
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
                    request, transformed_output, self._feature_stores
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

            results = self._worker.prepare_outputs(reply)
            response = MessageHandler.build_response(
                status=reply.status_enum,
                message=reply.message,
                result=results,
                custom_attributes=None,
            )

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing

        serialized_resp = MessageHandler.serialize_response(response)

        timings.append(time.perf_counter() - interm)  # timing
        interm = time.perf_counter()  # timing
        if request.callback:
            # send serialized response
            request.callback.send(serialized_resp)
            if reply.outputs:
                # send tensor data after response
                for output in reply.outputs:
                    request.callback.send(output)

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
