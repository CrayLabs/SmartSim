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

from ...infrastructure.storage.featurestore import FeatureStore
from ..worker.worker import MachineLearningWorkerBase
from .requestdispatcher import RequestBatch


class WorkerDevice:
    def __init__(self, name: str) -> None:
        """Wrapper around a device to keep track of loaded Models and availability
        :param name: name used by the toolkit to identify this device, e.g. ``cuda:0``
        """
        self._name = name
        """The name used by the toolkit to identify this device"""
        self._models: dict[str, t.Any] = {}
        """Dict of keys to models which are loaded on this device"""


    @property
    def name(self) -> str:
        """The identifier of the device represented by this object"""
        return self._name

    def add_model(self, key: str, model: t.Any) -> None:
        """Add a reference to a model loaded on this device and assign it a key

        :param key: The key under which the model is saved
        :param model: The model which is added
        """
        self._models[key] = model

    def remove_model(self, key: str) -> None:
        """Remove the reference to a model loaded on this device

        :param key: The key of the model to remove
        """
        self._models.pop(key)

    def get_model(self, key: str) -> t.Any:
        """Get the model corresponding to a given key

        :param key: the model key
        """
        return self._models[key]

    def __contains__(self, key: str) -> bool:
        return key in self._models


class DeviceManager:
    def __init__(self, device: WorkerDevice):
        self._device = device
        """Device managed by this object"""

    def _load_model_on_device(self,
        worker: MachineLearningWorkerBase,
        batch: RequestBatch,
        feature_store: t.Optional[FeatureStore],
    ) -> None:
        model_bytes = worker.fetch_model(batch, feature_store)
        loaded_model = worker.load_model(
            batch, model_bytes, self._device.name
        )
        self._device.add_model(batch.model_key, loaded_model.model)

    def get_device(
        self,
        worker: MachineLearningWorkerBase,
        batch: RequestBatch,
        feature_store: t.Optional[FeatureStore],
    ) -> t.Generator[WorkerDevice, None, None]:
        """Get the device managed by this object

        the model needed to run the batch of requests is
        guaranteed to be available on the model

        :param worker: The worker that wants to access the device
        :param batch: The batch of requests
        :param feature_store: The feature store on which part of the
        data needed by the request may be stored
        :return: A generator yielding the device
        """
        model_in_request = batch.has_raw_model

        # Load model if not already loaded, or
        # because it is sent with the request
        if model_in_request or not batch.model_key in self._device:
            self._load_model_on_device(worker, batch, feature_store)

        try:
            yield self._device
        finally:
            if model_in_request:
                self._device.remove_model(batch.model_key)
