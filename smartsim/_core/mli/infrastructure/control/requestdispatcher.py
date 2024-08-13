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
from dragon.managed_memory import MemoryPool
from dragon.mpbridge.queues import DragonQueue
import dragon.globalservices.pool as dragon_gs_pool

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
from ...comm.channel.channel import CommChannelBase
from ...comm.channel.dragonchannel import DragonCommChannel
from ...infrastructure.environmentloader import EnvironmentConfigLoader
from ...infrastructure.storage.featurestore import FeatureStore
from ...infrastructure.worker.worker import (
    RequestBatch,
    InferenceRequest,
    MachineLearningWorkerBase,
)
from ...message_handler import MessageHandler
from ...mli_schemas.model.model_capnp import Model
from ...mli_schemas.tensor.tensor_capnp import TensorDescriptor
from .commons import exception_handler

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status

logger = get_logger("Request Dispatcher")


def deserialize_message(
    data_blob: bytes,
    channel_type: t.Type[CommChannelBase],
) -> InferenceRequest:
    """Deserialize a message from a byte stream into an InferenceRequest
    :param data_blob: The byte stream to deserialize
    :param channel_type: The channel used to send the response"""

    request = MessageHandler.deserialize_request(data_blob)
    model_key: t.Optional[str] = None
    model_bytes: t.Optional[Model] = None

    if request.model.which() == "key":
        model_key = request.model.key.key
    elif request.model.which() == "data":
        model_bytes = request.model.data

    callback_key = request.replyChannel.descriptor

    # todo: shouldn't this be `CommChannel.find` instead of `DragonCommChannel`
    comm_channel = channel_type(callback_key)

    input_keys: t.Optional[t.List[str]] = None
    input_bytes: t.Optional[t.List[bytes]] = None

    output_keys: t.Optional[t.List[str]] = None

    input_meta: t.Optional[t.List[TensorDescriptor]] = None

    if request.input.which() == "keys":
        input_keys = [input_key.key for input_key in request.input.keys]
    elif request.input.which() == "descriptors":
        input_meta = request.input.descriptors  # type: ignore

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
        return self._lock.acquire(blocking=blocking, timeout=timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(
        self,
        exc_type: t.Optional[t.Type[BaseException]],
        exc_val: t.Optional[BaseException],
        exc_tb: t.Optional[TracebackType],
    ) -> None:
        self.release()


class BatchQueue(Queue[InferenceRequest]):
    def __init__(self, batch_timeout: float, batch_size: int, model_key: str) -> None:
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
        self._model_key = model_key
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
    def model_key(self) -> str:
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
        if self.empty():
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
        """Whether this queue can be deleted and garbafe collected"""
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
        comm_channel_type: t.Type[CommChannelBase] = DragonCommChannel,
    ) -> None:
        """The RequestDispatcher intercepts inference requests, stages them in
        queues and batches them together before making them available to Worker
        Managers.
        :param batch_timeout: Maximum elapsed time before flushing a complete or incomplete batch
        :param batch_size: Total capacity of each batch queue.
        :param mem_pool: Memory pool used to share batched input tensors with worker
        managers
        :param config_loader: Object to load configuration from environment
        :param worker_type: Type of worker to instantiate to batch inputs
        :param comm_channel_type: Type of channel used to get requests
        :raises SmartSimError: If config_loaded.get_queue() does not return a channel
        """
        super().__init__(as_service=True, cooldown=1)
        self._queues: dict[str, list[BatchQueue]] = []
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
        self._feature_store: t.Optional[FeatureStore] = (
            config_loader.get_feature_store()
        )
        """A feature store to retrieve models from"""
        self._comm_channel_type = comm_channel_type
        """The type of the channel used to receive requests"""
        self._worker = worker_type()
        """The worker used to batch inputs"""
        self._mem_pool = MemoryPool.attach(dragon_gs_pool.create(2 * 1024**3).sdesc)
        """Memory pool used to share batched input tensors with the Worker Managers"""
        self._perf_timer = PerfTimer(prefix="r_", debug=True, timing_on=True)
        """Performance timer"""

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

    def _on_start(self) -> None:
        self._queue_swap_lock = RLock()

    def _on_iteration(self) -> None:

        try:
            bytes_list: t.List[bytes] = self._incoming_channel.recv()
        except Exception:
            self._perf_timer.start_timings()
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

            request = deserialize_message(request_bytes, self._comm_channel_type)
            if request.input_meta and tensor_bytes_list:
                request.raw_inputs = tensor_bytes_list

            self._perf_timer.measure_time("deserialize_message")
            if not self._validate_request(request):
                return

            self._perf_timer.measure_time("validate_request")
            self.dispatch(request)

            self._perf_timer.measure_time("dispatch")
        finally:
            self.flush_requests()
            # TODO: implement this
            # self.remove_queues()

            self._perf_timer.end_timings()

        if self._perf_timer.max_length == 801:
            self._perf_timer.print_timings(True)

    @property
    def task_queue(self) -> DragonQueue:
        """The queue on which batched requests are placed"""
        return self._outgoing_queue

    def _swap_queue(self, model_key: str) -> None:
        """Get an empty queue or create a new one

        and make it the active one for a given model.

        :param model_key: The key of the model for which the
        queue has to be swapped
        :raises SmartSimError: If the queue is not locked.
        """
        if self._queue_swap_lock is None:
            raise SmartSimError("Queues were not locked")
        with self._queue_swap_lock:
            for queue_list in self._queues[model_key]:
                for queue in queue_list:
                    if not queue.full():
                        self._active_queues[model_key] = queue
                        return

            new_queue = BatchQueue(self._batch_timeout, self._batch_size, model_key)
            if model_key in self._queues:
                self._queues[model_key].append(new_queue)
            else:
                self._queues[model_key] = [new_queue]
            self._active_queues[model_key] = new_queue
            return

    def dispatch(self, request: InferenceRequest) -> None:
        """Assign a request to a batch queue
        :param request: the request to place
        """
        if request.raw_model is not None:
            logger.info("Direct inference requested, creating tmp queue")
            tmp_id = f"_tmp_{str(uuid.uuid4())}"
            tmp_queue: BatchQueue = BatchQueue(
                batch_timeout=0, batch_size=1, model_key=tmp_id
            )
            self._active_queues[tmp_id] = tmp_queue
            tmp_queue.put_nowait(request)
            tmp_queue.make_disposable()
            return

        if request.model_key:
            success = False
            while not success:
                try:
                    self._active_queues[request.model_key].put_nowait(request)
                    success = True
                except (Full, KeyError):
                    self._swap_queue(request.model_key)

    def flush_requests(self) -> None:
        """Get all requests from queues which are ready to be flushed. Place all
        avaliable request batches in the outgoing queue.
        """
        for queue_list in self._queues:
            for queue in queue_list:
                if queue.ready and queue.acquire(blocking=False):
                    self._perf_timer.measure_time("find_queue")
                    try:
                        batch = RequestBatch(
                            model_key=queue.model_key, requests=queue.flush(), inputs=None
                        )
                    finally:
                        self._perf_timer.measure_time("flush_requests")
                        queue.release()
                    try:
                        fetch_results = self._worker.fetch_inputs(
                            batch=batch, feature_store=self._feature_store
                        )
                    except Exception as exc:
                        exception_handler(
                            exc,
                            None,
                            "Error fetching input.",
                        )
                    self._perf_timer.measure_time("fetch_input")
                    try:
                        transformed_inputs = self._worker.transform_input(
                            batch=batch, fetch_results=fetch_results, mem_pool=self._mem_pool
                        )
                    except Exception as exc:
                        exception_handler(
                            exc,
                            None,
                            "Error Transforming input.",
                        )

                    self._perf_timer.measure_time("transform_input")
                    batch.inputs = transformed_inputs
                    for request in batch.requests:
                        request.raw_inputs = []
                        request.input_meta = []

                    self._outgoing_queue.put(batch)
                    self._perf_timer.measure_time("put")

    def _can_shutdown(self) -> bool:
        return False
