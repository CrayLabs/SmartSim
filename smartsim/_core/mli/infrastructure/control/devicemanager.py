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

import typing as t
from contextlib import contextmanager
from threading import RLock
from types import TracebackType

from ...infrastructure.storage.featurestore import FeatureStore
from ..worker.worker import MachineLearningWorkerBase
from .requestdispatcher import InferenceBatch


class WorkerDevice:
    def __init__(self, name: str) -> None:
        """Wrapper around a device to keep track of loaded Models and availability
        :param name: name used by the toolkit to identify this device, e.g. ``cuda:0``
        """
        self._name = name
        """The name used by the toolkit to identify this device"""
        self._lock = RLock()
        """Lock to ensure only one thread at the time accesses this device"""
        self._models: dict[str, t.Any] = {}

    def acquire(self, blocking: bool = True, timeout: float = -1) -> t.Optional[bool]:
        return self._lock.acquire(blocking=blocking, timeout=timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> None:
        self.acquire()

    @property
    def name(self) -> str:
        return self._name

    def add_model(self, key: str, model: t.Any) -> None:
        self._models[key] = model

    def remove_model(self, key: str) -> None:
        self._models.pop(key)

    def get_model(self, key: str) -> t.Any:
        return self._models[key]

    def __contains__(self, key: str) -> bool:
        return key in self._models

    def __exit__(
        self,
        exc_type: t.Optional[t.Type[BaseException]],
        exc_val: t.Optional[BaseException],
        exc_tb: t.Optional[TracebackType],
    ) -> None:
        self.release()


class DeviceManager:
    def __init__(self, devices: list[WorkerDevice]):
        self._devices = devices
        """Dictionary of model key to devices on which it is loaded"""

    def get_free_device(
        self,
        worker: MachineLearningWorkerBase,
        batch: InferenceBatch,
        feature_store: t.Optional[FeatureStore],
    ) -> t.Generator[WorkerDevice, None, None]:
        return_device = None
        sample_request = batch.requests[0]
        direct_inference = sample_request.raw_model is not None
        while return_device is None:
            loaded_devices = []
            if not direct_inference:
                # Look up devices to see if any of them already has a copy of the model
                for device in self._devices:
                    if batch.model_key in device:
                        loaded_devices.append(device)

                # If a pre-loaded model is found on a device, try using that device
                for device in loaded_devices:
                    if device.acquire(blocking=False):
                        return_device = device
                        break

            # If the model is not loaded on a free device, load it on another device (if available)
            if return_device is None:
                for candidate_device in self._devices:
                    if (
                        candidate_device not in loaded_devices
                        and candidate_device.acquire(blocking=False)
                    ):
                        model_bytes = worker.fetch_model(batch, feature_store)
                        loaded_model = worker.load_model(
                            batch, model_bytes, candidate_device.name
                        )
                        candidate_device.add_model(batch.model_key, loaded_model.model)

                        return_device = candidate_device

        try:
            yield return_device
        finally:
            if direct_inference:
                return_device.remove_model(batch.model_key)
            return_device.release()
