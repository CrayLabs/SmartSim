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
from queue import Empty

from smartsim._core.mli.infrastructure.storage.feature_store import FeatureStore

from .....log import get_logger
from ....entrypoints.service import Service
from ....utils.timings import PerfTimer
from ...message_handler import MessageHandler
from ..environment_loader import EnvironmentConfigLoader
from ..worker.worker import (
    InferenceReply,
    LoadModelResult,
    MachineLearningWorkerBase,
    RequestBatch,
)
from .device_manager import DeviceManager, WorkerDevice
from .error_handling import build_failure_reply, exception_handler

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status

logger = get_logger(__name__)


class WorkerManager(Service):
    """An implementation of a service managing distribution of tasks to
    machine learning workers."""

    def __init__(
        self,
        config_loader: EnvironmentConfigLoader,
        worker_type: t.Type[MachineLearningWorkerBase],
        dispatcher_queue: "mp.Queue[RequestBatch]",
        as_service: bool = False,
        cooldown: int = 0,
        device: t.Literal["cpu", "gpu"] = "cpu",
    ) -> None:
        """Initialize the WorkerManager.

        :param config_loader: Environment config loader for loading queues
        and feature stores
        :param worker_type: The type of worker to manage
        :param dispatcher_queue: Queue from which the batched requests are pulled
        :param as_service: Specifies run-once or run-until-complete behavior of service
        :param cooldown: Number of seconds to wait before shutting down after
        shutdown criteria are met
        :param device: The device on which the Worker should run. Every worker manager
        is assigned one single GPU (if available), thus the device should have no index.
        """
        super().__init__(as_service, cooldown)

        self._dispatcher_queue = dispatcher_queue
        """The Dispatcher queue that the WorkerManager monitors for new batches"""
        self._worker = worker_type()
        """The ML Worker implementation"""
        self._callback_factory = config_loader._callback_factory
        """The type of communication channel to construct for callbacks"""
        self._device = device
        """Device on which workers need to run"""
        self._cached_models: dict[str, t.Any] = {}
        """Dictionary of previously loaded models"""
        self._feature_stores: t.Dict[str, FeatureStore] = {}
        """A collection of attached feature stores"""
        self._featurestore_factory = config_loader._featurestore_factory
        """A factory method to create a desired feature store client type"""
        self._backbone: t.Optional[FeatureStore] = config_loader.get_backbone()
        """A standalone, system-created feature store used to share internal
        information among MLI components"""
        self._device_manager: t.Optional[DeviceManager] = None
        """Object responsible for model caching and device access"""
        self._perf_timer = PerfTimer(prefix="w_", debug=False, timing_on=True)
        """Performance timer"""

    @property
    def has_featurestore_factory(self) -> bool:
        """Check if the WorkerManager has a FeatureStore factory.

        :returns: True if there is a FeatureStore factory, False otherwise
        """
        return self._featurestore_factory is not None

    def _on_start(self) -> None:
        """Called on initial entry into Service `execute` event loop before
        `_on_iteration` is invoked."""
        self._device_manager = DeviceManager(WorkerDevice(self._device))

    def _check_feature_stores(self, batch: RequestBatch) -> bool:
        """Ensures that all feature stores required by the request are available.

        :param batch: The batch of requests to validate
        :returns: False if feature store validation fails for the batch, True otherwise
        """
        # collect all feature stores required by the request
        fs_model: t.Set[str] = set()
        if batch.model_id.key:
            fs_model = {batch.model_id.descriptor}
        fs_inputs = {key.descriptor for keys in batch.input_keys for key in keys}
        fs_outputs = {
            key.descriptor for keys in batch.output_key_refs.values() for key in keys
        }

        # identify which feature stores are requested and unknown
        fs_desired = fs_model.union(fs_inputs).union(fs_outputs)
        fs_actual = {item.descriptor for item in self._feature_stores.values()}
        fs_missing = fs_desired - fs_actual

        if not self.has_featurestore_factory:
            logger.error("No feature store factory configured")
            return False

        # create the feature stores we need to service request
        if fs_missing:
            logger.debug(f"Adding feature store(s): {fs_missing}")
            for descriptor in fs_missing:
                feature_store = self._featurestore_factory(descriptor)
                self._feature_stores[descriptor] = feature_store

        return True

    def _validate_batch(self, batch: RequestBatch) -> bool:
        """Ensure the request can be processed.

        :param batch: The batch of requests to validate
        :returns: False if the request fails any validation checks, True otherwise
        """
        if batch is None or not batch.has_callbacks:
            return False

        return self._check_feature_stores(batch)

    # remove this when we are done with time measurements
    # pylint: disable-next=too-many-statements
    def _on_iteration(self) -> None:
        """Executes calls to the machine learning worker implementation to complete
        the inference pipeline."""
        pre_batch_time = time.perf_counter()
        try:
            batch: RequestBatch = self._dispatcher_queue.get(timeout=0.0001)
        except Empty:
            return

        self._perf_timer.start_timings(
            "flush_requests", time.perf_counter() - pre_batch_time
        )

        if not self._validate_batch(batch):
            exception_handler(
                ValueError("An invalid batch was received"),
                None,
                None,
            )
            return

        if not self._device_manager:
            for callback_desc in batch.callback_descriptors:
                msg = "No Device Manager found. WorkerManager._on_start() "
                "must be called after initialization. If possible, "
                "you should use `WorkerManager.execute()` instead of "
                "directly calling `_on_iteration()`."
                try:
                    self._dispatcher_queue.put(batch)
                except Exception:
                    msg += "\nThe batch could not be put back in the queue "
                    "and will not be processed."
                exception_handler(
                    RuntimeError(msg),
                    self._callback_factory(callback_desc),
                    "Error acquiring device manager",
                )
            return

        try:
            device_cm = self._device_manager.get_device(
                worker=self._worker,
                batch=batch,
                feature_stores=self._feature_stores,
            )
        except Exception as exc:
            for callback_desc in batch.callback_descriptors:
                exception_handler(
                    exc,
                    self._callback_factory(callback_desc),
                    "Error loading model on device or getting device.",
                )
            return
        self._perf_timer.measure_time("fetch_model")

        with device_cm as device:

            try:
                model_result = LoadModelResult(device.get_model(batch.model_id.key))
            except Exception as exc:
                for callback_desc in batch.callback_descriptors:
                    exception_handler(
                        exc,
                        self._callback_factory(callback_desc),
                        "Error getting model from device.",
                    )
                return
            self._perf_timer.measure_time("load_model")

            if not batch.inputs:
                for callback_desc in batch.callback_descriptors:
                    exception_handler(
                        ValueError("Error batching inputs"),
                        self._callback_factory(callback_desc),
                        None,
                    )
                return
            transformed_input = batch.inputs

            try:
                execute_result = self._worker.execute(
                    batch, model_result, transformed_input, device.name
                )
            except Exception as e:
                for callback_desc in batch.callback_descriptors:
                    exception_handler(
                        e,
                        self._callback_factory(callback_desc),
                        "Error while executing.",
                    )
                return
            self._perf_timer.measure_time("execute")

            try:
                transformed_outputs = self._worker.transform_output(
                    batch, execute_result
                )
            except Exception as e:
                for callback_desc in batch.callback_descriptors:
                    exception_handler(
                        e,
                        self._callback_factory(callback_desc),
                        "Error while transforming the output.",
                    )
                return

            for callback_desc, transformed_output in zip(
                batch.callback_descriptors, transformed_outputs
            ):
                reply = InferenceReply()
                if batch.output_key_refs:
                    try:
                        output_keys = batch.output_key_refs[callback_desc]
                        reply.output_keys = self._worker.place_output(
                            output_keys,
                            transformed_output,
                            self._feature_stores,
                        )
                    except KeyError:
                        # the callback is not in the output_key_refs dict
                        # because it doesn't have output_keys associated with it
                        continue
                    except Exception as e:
                        exception_handler(
                            e,
                            self._callback_factory(callback_desc),
                            "Error while placing the output.",
                        )
                        continue
                else:
                    reply.outputs = transformed_output.outputs
                self._perf_timer.measure_time("assign_output")

                if not reply.has_outputs:
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

                self._perf_timer.measure_time("build_reply")

                serialized_resp = MessageHandler.serialize_response(response)

                self._perf_timer.measure_time("serialize_resp")

                callback = self._callback_factory(callback_desc)
                callback.send(serialized_resp)
                if reply.has_outputs:
                    for output in reply.outputs:
                        callback.send(output)
                self._perf_timer.measure_time("send")

        self._perf_timer.end_timings()

        if self._perf_timer.max_length == 801:
            self._perf_timer.print_timings(True)

    def _can_shutdown(self) -> bool:
        """Determine if the service can be shutdown.

        :returns: True when criteria to shutdown the service are met, False otherwise
        """
        # todo: determine shutdown criteria
        # will we receive a completion message?
        # will we let MLI mgr just kill this?
        # time_diff = self._last_event - datetime.datetime.now()
        # if time_diff.total_seconds() > self._cooldown:
        #     return True
        # return False
        return self._worker is None
