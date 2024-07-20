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
from dragon.mpbridge.queues import DragonQueue

# pylint: enable=import-error

# isort: off
# isort: on

import multiprocessing as mp
import time
import typing as t
import uuid
from queue import Empty, Full, Queue
from threading import Lock
from types import TracebackType

from packaging.version import Version

from .....error import SmartSimError
from .....log import get_logger
from ....utils.timings import PerfTimer
from ...comm.channel.channel import CommChannelBase
from ...comm.channel.dragonchannel import DragonCommChannel
from ...infrastructure.storage.featurestore import FeatureStore
from ...infrastructure.worker.torch_worker import TorchWorker
from ...infrastructure.worker.worker import InferenceBatch, InferenceRequest
from ...message_handler import MessageHandler
from ...mli_schemas.model.model_capnp import Model
from ...mli_schemas.response.response_capnp import ResponseBuilder
from ...mli_schemas.tensor.tensor_capnp import TensorDescriptor

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status

logger = get_logger("Request Dispatcher")


def deserialize_message(
    data_blob: bytes,
    channel_type: t.Type[CommChannelBase],
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


def build_failure_reply(status: "Status", message: str) -> ResponseBuilder:
    return MessageHandler.build_response(
        status=status,
        message=message,
        result=[],
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


class WorkerDevice:
    def __init__(self, name: str) -> None:
        """Wrapper around a device to keep track of loaded Models and availability
        :param name: name used by the toolkit to identify this device, e.g. ``cuda:0``
        """
        self._name = name
        """The name used by the toolkit to identify this device"""
        self._models: dict[str, t.Any] = {}
        """Dictionary of model key to model for models stored on this device"""
        self._lock = Lock()
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
        super().__init__(maxsize=batch_size)
        self._batch_timeout = batch_timeout
        self._batch_size = batch_size
        self._first_put: t.Optional[float] = None
        self._disposable = False
        self._model_key = model_key
        self._flush_lock = Lock()
        self._id = str(uuid.uuid4())

    @property
    def queue_id(self) -> str:
        return self._id

    def acquire(self, blocking: bool = True, timeout: float = -1) -> t.Optional[bool]:
        return self._flush_lock.acquire(blocking=blocking, timeout=timeout)

    def release(self) -> None:
        self._flush_lock.release()

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(
        self,
        exc_type: t.Optional[t.Type[BaseException]],
        exc_val: t.Optional[BaseException],
        exc_tb: t.Optional[TracebackType],
    ) -> None:
        self.release()

    @property
    def model_key(self) -> str:
        return self._model_key

    def put(
        self,
        item: InferenceRequest,
        block: bool = False,
        timeout: t.Optional[float] = 0.0,
    ) -> None:
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
    def _waited_time(self) -> float:
        if self._first_put is None:
            return 0
        return time.time() - self._first_put

    @property
    def ready(self) -> bool:
        if self.empty():
            return False
        return self.full() or (self._waited_time >= self._batch_timeout)

    def make_disposable(self) -> None:
        self._disposable = True

    @property
    def disposable(self) -> bool:
        return self.empty() and self._disposable

    def flush(self) -> list[t.Any]:
        num_items = self.qsize()
        self._first_put = None
        items = []
        # Avoid (unlikely) race condition error
        for _ in range(num_items):
            try:
                items.append(self.get())
            except Empty:
                break

        return items

    def full(self) -> bool:
        if self._disposable:
            return True
        if self._batch_size <= 0:
            return False
        return self.qsize() >= self._batch_size

    def empty(self) -> bool:
        return self.qsize() == 0


class RequestDispatcher:
    def __init__(
        self,
        batch_timeout: float,
        batch_size: int,
        incoming_channel: t.Optional[CommChannelBase],
        comm_channel_type: t.Type[CommChannelBase] = DragonCommChannel,
        feature_store: t.Optional[FeatureStore] = None,
    ) -> None:
        mp.set_start_method("dragon")
        self._queues: list[BatchQueue] = []
        self._active_queues: dict[str, BatchQueue] = {}
        self._model_last_version: dict[str, Version] = {}
        self._model_name_to_key: dict[str, str] = {}
        self._batch_timeout = batch_timeout
        self._batch_size = batch_size
        self._queue_swap_lock: t.Optional[Lock] = None
        self._incoming_channel = incoming_channel
        self._outgoing_queue: DragonQueue = mp.Queue(maxsize=0)
        self._feature_store = feature_store
        self._comm_channel_type = comm_channel_type
        self._perf_timer = PerfTimer(prefix="r_", debug=True)
        self._worker = TorchWorker()

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

    def run(self) -> None:
        self._queue_swap_lock = Lock()
        if self._incoming_channel is None:
            raise SmartSimError("No incoming channel for dispatcher")
        while True:
            try:
                bytes_list: t.List[bytes] = self._incoming_channel.recv()
            except Exception:
                pass
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
                    continue
                self._perf_timer.measure_time("validate_request")
                self.dispatch(request)
                self._perf_timer.measure_time("dispatch")
            finally:
                self.flush_requests()
                # TODO: implement this
                # self.remove_queues()

                self._perf_timer.end_timings()

                if self._perf_timer.max_length == 4*801:
                    self._perf_timer.print_timings(False)

    @property
    def task_queue(self) -> DragonQueue:
        return self._outgoing_queue

    def _swap_queue(self, model_key: str) -> None:
        if self._queue_swap_lock is None:
            raise SmartSimError("Queue was not locked")
        with self._queue_swap_lock:
            for queue in self._queues:
                if queue.model_key == model_key and not queue.full():
                    self._active_queues[model_key] = queue
                    return

            new_queue = BatchQueue(self._batch_timeout, self._batch_size, model_key)
            self._queues.append(new_queue)
            self._active_queues[model_key] = new_queue
            return

    def dispatch(self, request: InferenceRequest) -> None:
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

    def _update_model_version(self, model: Model) -> None:
        if not model.version:
            return
        if (
            model.name not in self._model_last_version
            or Version(model.version) > self._model_last_version[model.name]
        ):
            self._model_last_version[model.name] = Version(model.version)
            return

    def flush_requests(self) -> None:
        for queue in self._queues:
            if queue.ready and queue.acquire(blocking=False):
                self._perf_timer.measure_time("find_queue")
                try:
                    batch = InferenceBatch(
                        model_key=queue.model_key, requests=queue.flush(), inputs=None
                    )
                finally:
                    self._perf_timer.measure_time("flush_requests")
                    queue.release()
                fetch_results = self._worker.fetch_inputs(
                    batch=batch, feature_store=self._feature_store
                )
                self._perf_timer.measure_time("fetch_input")
                transformed_inputs = self._worker.transform_input(
                    batch=batch, fetch_results=fetch_results
                )
                self._perf_timer.measure_time("transform_input")
                batch.inputs = transformed_inputs
                for request in batch.requests:
                    request.raw_inputs = []
                    request.input_meta = []
                self._outgoing_queue.put(batch)
                self._perf_timer.measure_time("put")
