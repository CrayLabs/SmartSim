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
from contextlib import contextmanager, _GeneratorContextManager

from .....log import get_logger
from ...infrastructure.storage.featurestore import FeatureStore
from ..worker.worker import MachineLearningWorkerBase, RequestBatch

logger = get_logger(__name__)


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
        :returns: the model for the given key
        """
        return self._models[key]

    def __contains__(self, key: str) -> bool:
        """Check if model with a given key is available on the device

        :param key: the key of the model to check for existence
        :returns: whether the model is available on the device
        """
        return key in self._models

    @contextmanager
    def get(self, key_to_remove: t.Optional[str]) -> t.Iterator[t.Self]:
        yield self
        if key_to_remove is not None:
            self.remove_model(key_to_remove)


class DeviceManager:
    def __init__(self, device: WorkerDevice):
        """An object to manage devices such as GPUs and CPUs.

        The main goal of the ``DeviceManager`` is to ensure that
        the managed device is ready to be used by a worker to
        run a given model
        :param device: The managed device
        """
        self._device = device
        """Device managed by this object"""

    def _load_model_on_device(
        self,
        worker: MachineLearningWorkerBase,
        batch: RequestBatch,
        feature_stores: dict[str, FeatureStore],
    ) -> None:
        """Load the model needed to execute on a batch on the managed device.

        The model is loaded by the worker.

        :param worker: the worker that loads the model
        :param batch: the batch for which the model is needed
        :param feature_stores: feature stores where the model could be stored
        """

        model_bytes = worker.fetch_model(batch, feature_stores)
        loaded_model = worker.load_model(batch, model_bytes, self._device.name)
        self._device.add_model(batch.model_key.key, loaded_model.model)

    def get_device(
        self,
        worker: MachineLearningWorkerBase,
        batch: RequestBatch,
        feature_stores: dict[str, FeatureStore],
    ) -> _GeneratorContextManager[WorkerDevice]:
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
        if model_in_request or not batch.model_key.key in self._device:
            self._load_model_on_device(worker, batch, feature_stores)

        key_to_remove = batch.model_key.key if model_in_request else None
        return self._device.get(key_to_remove)
