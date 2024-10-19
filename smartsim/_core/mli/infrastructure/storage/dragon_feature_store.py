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

# pylint: disable=import-error
# isort: off
import dragon.data.ddict.ddict as dragon_ddict

# isort: on

from smartsim._core.mli.infrastructure.storage.dragon_util import (
    ddict_to_descriptor,
    descriptor_to_ddict,
)
from smartsim._core.mli.infrastructure.storage.feature_store import FeatureStore
from smartsim.error import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class DragonFeatureStore(FeatureStore):
    """A feature store backed by a dragon distributed dictionary."""

    def __init__(self, storage: "dragon_ddict.DDict") -> None:
        """Initialize the DragonFeatureStore instance.

        :param storage: A distributed dictionary to be used as the underlying
        storage mechanism of the feature store"""
        if storage is None:
            raise ValueError(
                "Storage is required when instantiating a DragonFeatureStore."
            )

        descriptor = ""
        if isinstance(storage, dragon_ddict.DDict):
            descriptor = ddict_to_descriptor(storage)

        super().__init__(descriptor)
        self._storage: t.Dict[str, t.Union[str, bytes]] = storage
        """The underlying storage mechanism of the DragonFeatureStore; a
        distributed, in-memory key-value store"""

    def _get(self, key: str) -> t.Union[str, bytes]:
        """Retrieve a value from the underlying storage mechanism.

        :param key: The unique key that identifies the resource
        :returns: The value identified by the key
        :raises KeyError: If the key has not been used to store a value
        """
        try:
            return self._storage[key]
        except dragon_ddict.DDictError as e:
            raise KeyError(f"Key not found in FeatureStore: {key}") from e

    def _set(self, key: str, value: t.Union[str, bytes]) -> None:
        """Store a value into the underlying storage mechanism.

        :param key: The unique key that identifies the resource
        :param value: The value to store
        :returns: The value identified by the key
        """
        self._storage[key] = value

    def _contains(self, key: str) -> bool:
        """Determine if the storage mechanism contains a given key.

        :param key: The unique key that identifies the resource
        :returns: True if the key is defined, False otherwise
        """
        return key in self._storage

    def pop(self, key: str) -> t.Union[str, bytes, None]:
        """Remove the value from the dictionary and return the value.

        :param key: Dictionary key to retrieve
        :returns: The value held at the key if it exists, otherwise `None
        `"""
        try:
            return self._storage.pop(key)
        except dragon_ddict.DDictError:
            return None

    @classmethod
    def from_descriptor(
        cls,
        descriptor: str,
    ) -> "DragonFeatureStore":
        """A factory method that creates an instance from a descriptor string.

        :param descriptor: The descriptor that uniquely identifies the resource
        :returns: An attached DragonFeatureStore
        :raises SmartSimError: If attachment to DragonFeatureStore fails
        """
        try:
            logger.debug(f"Attaching to FeatureStore with descriptor: {descriptor}")
            storage = descriptor_to_ddict(descriptor)
            return cls(storage)
        except Exception as ex:
            raise SmartSimError(
                f"Error creating dragon feature store from descriptor: {descriptor}"
            ) from ex
