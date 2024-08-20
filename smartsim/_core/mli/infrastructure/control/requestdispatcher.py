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
from threading import RLock
from types import TracebackType

from smartsim._core.entrypoints.service import Service

from .....error import SmartSimError
from .....log import get_logger
from ....utils.timings import PerfTimer
from ...infrastructure.environmentloader import EnvironmentConfigLoader
from ...infrastructure.storage.featurestore import FeatureStore, FeatureStoreKey
from ...infrastructure.worker.worker import (
    InferenceRequest,
    MachineLearningWorkerBase,
    RequestBatch,
)
from .commons import exception_handler

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status

logger = get_logger("Request Dispatcher")

# Placeholder
ModelIdentifier = FeatureStoreKey


class WorkerDevice:
    def __init__(self, name: str) -> None:
        """Wrapper around a device to keep track of loaded Models and availability
        :param name: name used by the toolkit to identify this device, e.g. ``cuda:0``
        """
        self._name = name
        """The name used by the toolkit to identify this device"""
        self._models: dict[str, t.Any] = {}
        """Dictionary of model key to model for models stored on this device"""
        self._lock = RLock()
        """Lock to ensure only one thread at the time accesses this device"""

    def acquire(self, blocking: bool = True, timeout: float = -1) -> t.Optional[bool]:
        """Acquire and lock this device to prevent other threads

        from acquiring it concurrently.
        :param blocking: If set to True, the call will block
        for the time specified by ``timeout`` until the lock
        can be acquired
        :param timeout: Time (in seconds) to wait to acquire lock.
        Ignored if ``blocking`` is set to False.
        """
        return self._lock.acquire(blocking=blocking, timeout=timeout)

    def release(self) -> None:
        """Release device to allow other threads to acquire it"""
        self._lock.release()

    def __enter__(self) -> None:
        """Locked context creator for this device"""
        self.acquire()

    def __exit__(
        self,
        exc_type: t.Optional[t.Type[BaseException]],
        exc_val: t.Optional[BaseException],
        exc_tb: t.Optional[TracebackType],
    ) -> None:
        """Locked context destructor for this device"""
        self.release()


class BatchQueue(Queue[InferenceRequest]):
    def __init__(
        self, batch_timeout: float, batch_size: int, model_key: ModelIdentifier
    ) -> None:
        """Queue used to store inference requests waiting to be batched and
        sent to Worker Managers.
        :param batch_timeout: Time in seconds that has to be waited before flushing a
        non-full queue. The time of the first item put is 0 seconds.
        :param batch_size: Total capacity of the queue.
        :param model_key: Key of the model which needs to be executed on the queued
        requests
        """
        super().__init__(maxsize=batch_size)
        self._batch_timeout = batch_timeout
        """Time in seconds that has to be waited before flushing a non-full queue.
        The time of the first item put is 0 seconds."""
        self._batch_size = batch_size
        """Total capacity of the queue."""
        self._first_put: t.Optional[float] = None
        """Time at which the first item was put on the queue"""
        self._disposable = False
        """Whether the queue will not be used again and can be deleted.
        A disposable queue is always full."""
        self._model_key: FeatureStoreKey = model_key
        """Key of the model which needs to be executed on the queued requets"""
        self._flush_lock = RLock()
        """Lock used to make sure only one process can flush the queue (unused now)"""
        self._uid = str(uuid.uuid4())
        """Unique ID of queue"""

    @property
    def uid(self) -> str:
        """ID of this queue"""
        return self._uid

    def acquire(self, blocking: bool = True, timeout: float = -1) -> t.Optional[bool]:
        """Acquire queue lock to flush
        :param blocking: whether to block on lock acquisition
        :param timeout: Time to wait if blocking, before raising exception
        """
        return self._flush_lock.acquire(blocking=blocking, timeout=timeout)

    def release(self) -> None:
        """Release queue lock"""
        self._flush_lock.release()

    def __enter__(self) -> None:
        """Method to use the Queue as a Context Manager"""
        self.acquire()

    def __exit__(
        self,
        exc_type: t.Optional[t.Type[BaseException]],
        exc_val: t.Optional[BaseException],
        exc_tb: t.Optional[TracebackType],
    ) -> None:
        """Method to release the Queue as a Context Manager"""
        self.release()

    @property
    def model_key(self) -> ModelIdentifier:
        """Key of the model which needs to be run on the queued requests"""
        return self._model_key

    def put(
        self,
        item: InferenceRequest,
        block: bool = False,
        timeout: t.Optional[float] = 0.0,
    ) -> None:
        """Put an inference request in the queue
        :param item: The request
        :param block: Whether to block when trying to put the item
        :param timeout: Time (in seconds) to wait if block==True
        :raises Full: If an item cannot be put on the queue
        """
        if not self.acquire(blocking=False):
            raise Full
        try:
            if self.full():
                raise Full
            if self._first_put is None:
                self._first_put = time.time()
            super().put(item, block=block, timeout=timeout)
        finally:
            self.release()

    @property
    def _elapsed_time(self) -> float:
        """Time elapsed since the first item was put on this queue"""
        if self.empty() or self._first_put is None:
            return 0
        return time.time() - self._first_put

    @property
    def ready(self) -> bool:
        """True if the queue can be flushed"""
        if self.empty():
            return False
        return self.full() or (self._elapsed_time >= self._batch_timeout)

    def make_disposable(self) -> None:
        """Set this queue as disposable, and never use it again after it gets flushed"""
        self._disposable = True

    @property
    def can_be_removed(self) -> bool:
        """Whether this queue can be deleted and garbage collected"""
        return self.empty() and self._disposable

    def flush(self) -> list[t.Any]:
        """Get all requests from queue
        :return: Requests waiting to be executed
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
        """Return True if the queue has reached its maximum capacity"""
        if self._disposable:
            return True
        if self._batch_size <= 0:
            return False
        return self.qsize() >= self._batch_size

    def empty(self) -> bool:
        """Return True if the queue has 0 elements"""
        return self.qsize() == 0


class RequestDispatcher(Service):
    def __init__(
        self,
        batch_timeout: float,
        batch_size: int,
        config_loader: EnvironmentConfigLoader,
        worker_type: t.Type[MachineLearningWorkerBase],
    ) -> None:
        """The RequestDispatcher intercepts inference requests, stages them in
        queues and batches them together before making them available to Worker
        Managers.
        :param batch_timeout: Maximum elapsed time before flushing a complete or
        incomplete batch
        :param batch_size: Total capacity of each batch queue.
        :param mem_pool: Memory pool used to share batched input tensors with worker
        managers
        :param config_loader: Object to load configuration from environment
        :param worker_type: Type of worker to instantiate to batch inputs
        :raises SmartSimError: If config_loaded.get_queue() does not return a channel
        """
        super().__init__(as_service=True, cooldown=1)
        self._queues: dict[str, list[BatchQueue]] = {}
        """Dict of all batch queues available for a given model key"""
        self._active_queues: dict[str, BatchQueue] = {}
        """Mapping telling which queue is the recipient of requests for a given model
        key"""
        self._batch_timeout = batch_timeout
        """Time in seconds that has to be waited before flushing a non-full queue"""
        self._batch_size = batch_size
        """Total capacity of each batch queue."""
        self._queue_swap_lock: t.Optional[RLock] = None
        """Lock used to swap the active queue for a key"""
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
        self._mem_pool = MemoryPool.attach(dragon_gs_pool.create(2 * 1024**3).sdesc)
        """Memory pool used to share batched input tensors with the Worker Managers"""
        self._perf_timer = PerfTimer(prefix="r_", debug=False, timing_on=True)
        """Performance timer"""

    def _check_feature_stores(self, request: InferenceRequest) -> bool:
        """Ensures that all feature stores required by the request are available

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

        if self._featurestore_factory is None:
            logger.error("No feature store factory configured")
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
        """Ensure that a model is available for the request

        :param request: The request to validate
        :returns: False if model validation fails for the request, True otherwise
        """
        if request.model_key or request.raw_model:
            return True

        logger.error("Unable to continue without model bytes or feature store key")
        return False

    # pylint: disable-next=no-self-use
    def _check_inputs(self, request: InferenceRequest) -> bool:
        """Ensure that inputs are available for the request

        :param request: The request to validate
        :returns: False if input validation fails for the request, True otherwise
        """
        if request.input_keys or request.raw_inputs:
            return True

        logger.error("Unable to continue without input bytes or feature store keys")
        return False

    # pylint: disable-next=no-self-use
    def _check_callback(self, request: InferenceRequest) -> bool:
        """Ensure that a callback channel is available for the request

        :param request: The request to validate
        :returns: False if callback validation fails for the request, True otherwise
        """
        if request.callback is not None:
            return True

        logger.error("No callback channel provided in request")
        return False

    def _validate_request(self, request: InferenceRequest) -> bool:
        """Ensure the request can be processed

        :param request: The request to validate
        :return: False if the request fails any validation checks, True otherwise"""
        checks = [
            self._check_feature_stores(request),
            self._check_model(request),
            self._check_inputs(request),
            self._check_callback(request),
        ]

        return all(checks)

    def _on_start(self) -> None:
        self._queue_swap_lock = RLock()

    def _on_iteration(self) -> None:
        try:
            self._perf_timer.set_active(True)
            bytes_list: t.List[bytes] = self._incoming_channel.recv()
        except Exception:
            self._perf_timer.set_active(False)
        else:
            if not bytes_list:
                exception_handler(
                    ValueError("No request data found"),
                    None,
                    "No request data found.",
                )

            request_bytes = bytes_list[0]
            tensor_bytes_list = bytes_list[1:]
            self._perf_timer.start_timings()

            request = self._worker.deserialize_message(
                request_bytes, self._callback_factory
            )
            if request.input_meta and tensor_bytes_list:
                request.raw_inputs = tensor_bytes_list

            self._perf_timer.measure_time("deserialize_message")

            if not self._validate_request(request):
                exception_handler(
                    ValueError("Error validating the request"),
                    request.callback,
                    "Error validating the request.",
                )
                self._perf_timer.measure_time("validate_request")
            else:
                self._perf_timer.measure_time("validate_request")
                self.dispatch(request)
                self._perf_timer.measure_time("dispatch")
        finally:
            self.flush_requests()
            # TODO: implement this
            # self.remove_queues()

            self._perf_timer.end_timings()

        if self._perf_timer.max_length == 801 and self._perf_timer.is_active:
            self._perf_timer.print_timings(True)

    @property
    def task_queue(self) -> DragonQueue:
        """The queue on which batched requests are placed"""
        return self._outgoing_queue

    def _swap_queue(self, model_key: FeatureStoreKey) -> None:
        """Get an empty queue or create a new one

        and make it the active one for a given model.
        :param model_key: The key of the model for which the
        queue has to be swapped
        :raises SmartSimError: If the queue is not locked.
        """
        if self._queue_swap_lock is None:
            raise SmartSimError("Queues were not locked")
        with self._queue_swap_lock:
            if model_key.key in self._queues:
                for queue in self._queues[model_key.key]:
                    if not queue.full():
                        self._active_queues[model_key.key] = queue
                        return

            new_queue = BatchQueue(self._batch_timeout, self._batch_size, model_key)
            if model_key.key in self._queues:
                self._queues[model_key.key].append(new_queue)
            else:
                self._queues[model_key.key] = [new_queue]
            self._active_queues[model_key.key] = new_queue
            return

    def dispatch(self, request: InferenceRequest) -> None:
        """Assign a request to a batch queue
        :param request: the request to place
        """
        if request.raw_model is not None:
            logger.debug("Direct inference requested, creating tmp queue")
            tmp_id = f"_tmp_{str(uuid.uuid4())}"
            tmp_queue: BatchQueue = BatchQueue(
                batch_timeout=0,
                batch_size=1,
                model_key=FeatureStoreKey(key=tmp_id, descriptor="TMP"),
            )
            self._active_queues[tmp_id] = tmp_queue
            tmp_queue.put_nowait(request)
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
        avaliable request batches in the outgoing queue.
        """
        for queue_list in self._queues.values():
            for queue in queue_list:
                if queue.ready and queue.acquire(blocking=False):
                    self._perf_timer.measure_time("find_queue")
                    try:
                        batch = RequestBatch(
                            requests=queue.flush(),
                            inputs=None,
                            model_key=queue.model_key,
                        )
                    finally:
                        self._perf_timer.measure_time("flush_requests")
                        queue.release()
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
                            "Error Transforming input.",
                        )
                        continue

                    self._perf_timer.measure_time("transform_input")
                    batch.inputs = transformed_inputs
                    for request in batch.requests:
                        request.raw_inputs = []
                        request.input_meta = []

                    self._outgoing_queue.put(batch)
                    self._perf_timer.measure_time("put")

    def _can_shutdown(self) -> bool:
        return False
