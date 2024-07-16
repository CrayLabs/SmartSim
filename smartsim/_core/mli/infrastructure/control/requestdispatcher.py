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
import uuid
from queue import Empty, Full, Queue
from threading import RLock
from types import TracebackType

from packaging.version import Version

from ...infrastructure.worker.worker import InferenceBatch, InferenceRequest
from ...mli_schemas.model.model_capnp import Model

if t.TYPE_CHECKING:
    from dragon.fli import FLInterface


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
        super().__init__(maxsize=batch_size)
        self._batch_timeout = batch_timeout
        self._batch_size = batch_size
        self._first_put: t.Optional[float] = None
        self._disposable = False
        self._model_key = model_key
        self._flush_lock = RLock()

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
        if not self.acquire(blocking=False) or self.disposable:
            raise Full
        if self._first_put is None:
            self._first_put = time.time()
        super().put(item, block=block, timeout=timeout)

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
        return self.qsize() >= self._batch_size

    def empty(self) -> bool:
        return self.qsize() == 0


class RequestDispatcher:
    def __init__(
        self,
        batch_timeout: float,
        batch_size: int,
    ) -> None:
        self._queues: list[BatchQueue]
        self._active_queues: dict[str, BatchQueue] = {}
        self._model_last_version: dict[str, Version] = {}
        self._model_name_to_key: dict[str, str] = {}
        self._batch_timeout = batch_timeout
        self._batch_size = batch_size
        self._queue_swap_lock = RLock()

    def _swap_queue(self, model_key: str) -> None:
        with self._queue_swap_lock:
            for queue in self._queues:
                if queue.model_key == model_key and not queue.full():
                    self._active_queues[model_key] = queue
                    return

            new_queue = BatchQueue(self._batch_timeout, self._batch_size, model_key)
            self._active_queues[model_key] = new_queue
            return

    def dispatch(self, request: InferenceRequest) -> None:
        if request.raw_model is not None:
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

    def flush_requests(self) -> t.Optional[InferenceBatch]:
        result = None
        for queue in self._queues:
            if queue.acquire(blocking=False) and queue.ready:
                result = InferenceBatch(
                    model_key=queue.model_key, requests=queue.flush()
                )
                queue.release()
                break

        return result
