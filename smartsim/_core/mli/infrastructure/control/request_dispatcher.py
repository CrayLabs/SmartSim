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
import dragon.globalservices.pool as dragon_gs_pool
from dragon.managed_memory import MemoryPool
from dragon.mpbridge.queues import DragonQueue

# pylint: enable=import-error

# isort: off
# isort: on

import multiprocessing as mp
import time
import typing as t
import uuid
from queue import Empty, Full, Queue

from smartsim._core.entrypoints.service import Service

from .....error import SmartSimError
from .....log import get_logger
from ....utils.timings import PerfTimer
from ..environment_loader import EnvironmentConfigLoader
from ..storage.feature_store import FeatureStore
from ..worker.worker import (
    InferenceRequest,
    MachineLearningWorkerBase,
    ModelIdentifier,
    RequestBatch,
)
from .error_handling import exception_handler

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status

logger = get_logger("Request Dispatcher")


class BatchQueue(Queue[InferenceRequest]):
    def __init__(
        self, batch_timeout: float, batch_size: int, model_id: ModelIdentifier
    ) -> None:
        """Queue used to store inference requests waiting to be batched and
        sent to Worker Managers.

        :param batch_timeout: Time in seconds that has to be waited before flushing a
        non-full queue. The time of the first item put is 0 seconds.
        :param batch_size: Total capacity of the queue
        :param model_id: Key of the model which needs to be executed on the queued
        requests
        """
        super().__init__(maxsize=batch_size)
        self._batch_timeout = batch_timeout
        """Time in seconds that has to be waited before flushing a non-full queue.
        The time of the first item put is 0 seconds."""
        self._batch_size = batch_size
        """Total capacity of the queue"""
        self._first_put: t.Optional[float] = None
        """Time at which the first item was put on the queue"""
        self._disposable = False
        """Whether the queue will not be used again and can be deleted.
        A disposable queue is always full."""
        self._model_id: ModelIdentifier = model_id
        """Key of the model which needs to be executed on the queued requests"""
        self._uid = str(uuid.uuid4())
        """Unique ID of queue"""

    @property
    def uid(self) -> str:
        """ID of this queue.

        :returns: Queue ID
        """
        return self._uid

    @property
    def model_id(self) -> ModelIdentifier:
        """Key of the model which needs to be run on the queued requests.

        :returns: Model key
        """
        return self._model_id

    def put(
        self,
        item: InferenceRequest,
        block: bool = False,
        timeout: t.Optional[float] = 0.0,
    ) -> None:
        """Put an inference request in the queue.

        :param item: The request
        :param block: Whether to block when trying to put the item
        :param timeout: Time (in seconds) to wait if block==True
        :raises Full: If an item cannot be put on the queue
        """
        super().put(item, block=block, timeout=timeout)
        if self._first_put is None:
            self._first_put = time.time()

    @property
    def _elapsed_time(self) -> float:
        """Time elapsed since the first item was put on this queue.

        :returns: Time elapsed
        """
        if self.empty() or self._first_put is None:
            return 0
        return time.time() - self._first_put

    @property
    def ready(self) -> bool:
        """Check if the queue can be flushed.

        :returns: True if the queue can be flushed, False otherwise
        """
        if self.empty():
            logger.debug("Request dispatcher queue is empty")
            return False

        timed_out = False
        if self._batch_timeout >= 0:
            timed_out = self._elapsed_time >= self._batch_timeout

        if self.full():
            logger.debug("Request dispatcher ready to deliver full batch")
            return True

        if timed_out:
            logger.debug("Request dispatcher delivering partial batch")
            return True

        return False

    def make_disposable(self) -> None:
        """Set this queue as disposable, and never use it again after it gets
        flushed."""
        self._disposable = True

    @property
    def can_be_removed(self) -> bool:
        """Determine whether this queue can be deleted and garbage collected.

        :returns: True if queue can be removed, False otherwise
        """
        return self.empty() and self._disposable

    def flush(self) -> list[InferenceRequest]:
        """Get all requests from queue.

        :returns: Requests waiting to be executed
        """
        num_items = self.qsize()
        self._first_put = None
        items = []
        for _ in range(num_items):
            try:
                items.append(self.get())
            except Empty:
                break

        return items

    def full(self) -> bool:
        """Check if the queue has reached its maximum capacity.

        :returns: True if the queue has reached its maximum capacity,
        False otherwise
        """
        if self._disposable:
            return True
        return self.qsize() >= self._batch_size

    def empty(self) -> bool:
        """Check if the queue is empty.

        :returns: True if the queue has 0 elements, False otherwise
        """
        return self.qsize() == 0


class RequestDispatcher(Service):
    def __init__(
        self,
        batch_timeout: float,
        batch_size: int,
        config_loader: EnvironmentConfigLoader,
        worker_type: t.Type[MachineLearningWorkerBase],
        mem_pool_size: int = 2 * 1024**3,
    ) -> None:
        """The RequestDispatcher intercepts inference requests, stages them in
        queues and batches them together before making them available to Worker
        Managers.

        :param batch_timeout: Maximum elapsed time before flushing a complete or
        incomplete batch
        :param batch_size: Total capacity of each batch queue
        :param mem_pool: Memory pool used to share batched input tensors with worker
        managers
        :param config_loader: Object to load configuration from environment
        :param worker_type: Type of worker to instantiate to batch inputs
        :param mem_pool_size: Size of the memory pool used to allocate tensors
        """
        super().__init__(as_service=True, cooldown=1)
        self._queues: dict[str, list[BatchQueue]] = {}
        """Dict of all batch queues available for a given model id"""
        self._active_queues: dict[str, BatchQueue] = {}
        """Mapping telling which queue is the recipient of requests for a given model
        key"""
        self._batch_timeout = batch_timeout
        """Time in seconds that has to be waited before flushing a non-full queue"""
        self._batch_size = batch_size
        """Total capacity of each batch queue"""
        incoming_channel = config_loader.get_queue()
        if incoming_channel is None:
            raise SmartSimError("No incoming channel for dispatcher")
        self._incoming_channel = incoming_channel
        """The channel the dispatcher monitors for new tasks"""
        self._outgoing_queue: DragonQueue = mp.Queue(maxsize=0)
        """The queue on which batched inference requests are placed"""
        self._feature_stores: t.Dict[str, FeatureStore] = {}
        """A collection of attached feature stores"""
        self._featurestore_factory = config_loader._featurestore_factory
        """A factory method to create a desired feature store client type"""
        self._backbone: t.Optional[FeatureStore] = config_loader.get_backbone()
        """A standalone, system-created feature store used to share internal
        information among MLI components"""
        self._callback_factory = config_loader._callback_factory
        """The type of communication channel to construct for callbacks"""
        self._worker = worker_type()
        """The worker used to batch inputs"""
        self._mem_pool = MemoryPool.attach(dragon_gs_pool.create(mem_pool_size).sdesc)
        """Memory pool used to share batched input tensors with the Worker Managers"""
        self._perf_timer = PerfTimer(prefix="r_", debug=False, timing_on=True)
        """Performance timer"""

    @property
    def has_featurestore_factory(self) -> bool:
        """Check if the RequestDispatcher has a FeatureStore factory.

        :returns: True if there is a FeatureStore factory, False otherwise
        """
        return self._featurestore_factory is not None

    def _check_feature_stores(self, request: InferenceRequest) -> bool:
        """Ensures that all feature stores required by the request are available.

        :param request: The request to validate
        :returns: False if feature store validation fails for the request, True
        otherwise
        """
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

        if not self.has_featurestore_factory:
            logger.error("No feature store factory is configured. Unable to dispatch.")
            return False

        # create the feature stores we need to service request
        if fs_missing:
            logger.debug(f"Adding feature store(s): {fs_missing}")
            for descriptor in fs_missing:
                feature_store = self._featurestore_factory(descriptor)
                self._feature_stores[descriptor] = feature_store

        return True

    # pylint: disable-next=no-self-use
    def _check_model(self, request: InferenceRequest) -> bool:
        """Ensure that a model is available for the request.

        :param request: The request to validate
        :returns: False if model validation fails for the request, True otherwise
        """
        if request.has_model_key or request.has_raw_model:
            return True

        logger.error("Unable to continue without model bytes or feature store key")
        return False

    # pylint: disable-next=no-self-use
    def _check_inputs(self, request: InferenceRequest) -> bool:
        """Ensure that inputs are available for the request.

        :param request: The request to validate
        :returns: False if input validation fails for the request, True otherwise
        """
        if request.has_input_keys or request.has_raw_inputs:
            return True

        logger.error("Unable to continue without input bytes or feature store keys")
        return False

    # pylint: disable-next=no-self-use
    def _check_callback(self, request: InferenceRequest) -> bool:
        """Ensure that a callback channel is available for the request.

        :param request: The request to validate
        :returns: False if callback validation fails for the request, True otherwise
        """
        if request.callback_desc:
            return True

        logger.error("No callback channel provided in request")
        return False

    def _validate_request(self, request: InferenceRequest) -> bool:
        """Ensure the request can be processed.

        :param request: The request to validate
        :returns: False if the request fails any validation checks, True otherwise
        """
        checks = [
            self._check_feature_stores(request),
            self._check_model(request),
            self._check_inputs(request),
            self._check_callback(request),
        ]

        return all(checks)

    def _on_iteration(self) -> None:
        """This method is executed repeatedly until ``Service`` shutdown
        conditions are satisfied and cooldown is elapsed."""
        try:
            self._perf_timer.is_active = True
            bytes_list: t.List[bytes] = self._incoming_channel.recv()
        except Exception:
            self._perf_timer.is_active = False
        else:
            if not bytes_list:
                exception_handler(
                    ValueError("No request data found"),
                    None,
                    None,
                )

            logger.debug(f"Dispatcher is processing {len(bytes_list)} messages")
            request_bytes = bytes_list[0]
            tensor_bytes_list = bytes_list[1:]
            self._perf_timer.start_timings()

            request = self._worker.deserialize_message(request_bytes)
            if request.has_input_meta and tensor_bytes_list:
                request.raw_inputs = tensor_bytes_list

            self._perf_timer.measure_time("deserialize_message")

            if not self._validate_request(request):
                exception_handler(
                    ValueError("Error validating the request"),
                    (
                        self._callback_factory(request.callback_desc)
                        if request.callback_desc
                        else None
                    ),
                    None,
                )
                self._perf_timer.measure_time("validate_request")
            else:
                self._perf_timer.measure_time("validate_request")
                self.dispatch(request)
                self._perf_timer.measure_time("dispatch")
        finally:
            self.flush_requests()
            self.remove_queues()

            self._perf_timer.end_timings()

        if self._perf_timer.max_length == 801 and self._perf_timer.is_active:
            self._perf_timer.print_timings(True)

    def remove_queues(self) -> None:
        """Remove references to queues that can be removed
        and allow them to be garbage collected."""
        queue_lists_to_remove = []
        for key, queues in self._queues.items():
            queues_to_remove = []
            for queue in queues:
                if queue.can_be_removed:
                    queues_to_remove.append(queue)

            for queue_to_remove in queues_to_remove:
                queues.remove(queue_to_remove)
                if (
                    key in self._active_queues
                    and self._active_queues[key] == queue_to_remove
                ):
                    del self._active_queues[key]

            if len(queues) == 0:
                queue_lists_to_remove.append(key)

        for key in queue_lists_to_remove:
            del self._queues[key]

    @property
    def task_queue(self) -> DragonQueue:
        """The queue on which batched requests are placed.

        :returns: The queue
        """
        return self._outgoing_queue

    def _swap_queue(self, model_id: ModelIdentifier) -> None:
        """Get an empty queue or create a new one
        and make it the active one for a given model.

        :param model_id: The id of the model for which the
        queue has to be swapped
        """
        if model_id.key in self._queues:
            for queue in self._queues[model_id.key]:
                if not queue.full():
                    self._active_queues[model_id.key] = queue
                    return

        new_queue = BatchQueue(self._batch_timeout, self._batch_size, model_id)
        if model_id.key in self._queues:
            self._queues[model_id.key].append(new_queue)
        else:
            self._queues[model_id.key] = [new_queue]
        self._active_queues[model_id.key] = new_queue
        return

    def dispatch(self, request: InferenceRequest) -> None:
        """Assign a request to a batch queue.

        :param request: The request to place
        """
        if request.has_raw_model:
            logger.debug("Direct inference requested, creating tmp queue")
            tmp_id = f"_tmp_{str(uuid.uuid4())}"
            tmp_queue: BatchQueue = BatchQueue(
                batch_timeout=0,
                batch_size=1,
                model_id=ModelIdentifier(key=tmp_id, descriptor="TMP"),
            )
            self._active_queues[tmp_id] = tmp_queue
            self._queues[tmp_id] = [tmp_queue]
            tmp_queue.put(request)
            tmp_queue.make_disposable()
            return

        if request.model_key:
            success = False
            while not success:
                try:
                    self._active_queues[request.model_key.key].put_nowait(request)
                    success = True
                except (Full, KeyError):
                    self._swap_queue(request.model_key)

    def flush_requests(self) -> None:
        """Get all requests from queues which are ready to be flushed. Place all
        available request batches in the outgoing queue."""
        for queue_list in self._queues.values():
            for queue in queue_list:
                if queue.ready:
                    self._perf_timer.measure_time("find_queue")
                    try:
                        batch = RequestBatch.from_requests(
                            queue.flush(), queue.model_id
                        )
                    finally:
                        self._perf_timer.measure_time("flush_requests")
                    try:
                        fetch_results = self._worker.fetch_inputs(
                            batch=batch, feature_stores=self._feature_stores
                        )
                    except Exception as exc:
                        exception_handler(
                            exc,
                            None,
                            "Error fetching input.",
                        )
                        continue
                    self._perf_timer.measure_time("fetch_input")
                    try:
                        transformed_inputs = self._worker.transform_input(
                            batch=batch,
                            fetch_results=fetch_results,
                            mem_pool=self._mem_pool,
                        )
                    except Exception as exc:
                        exception_handler(
                            exc,
                            None,
                            "Error transforming input.",
                        )
                        continue

                    self._perf_timer.measure_time("transform_input")
                    batch.inputs = transformed_inputs

                    try:
                        self._outgoing_queue.put(batch)
                    except Exception as exc:
                        exception_handler(
                            exc,
                            None,
                            "Error placing batch on task queue.",
                        )
                        continue
                    self._perf_timer.measure_time("put")

    def _can_shutdown(self) -> bool:
        """Determine whether the Service can be shut down.

        :returns: False
        """
        return False

    def __del__(self) -> None:
        """Destroy allocated memory resources."""
        # pool may be null if a failure occurs prior to successful attach
        pool: t.Optional[MemoryPool] = getattr(self, "_mem_pool", None)

        if pool:
            pool.destroy()
