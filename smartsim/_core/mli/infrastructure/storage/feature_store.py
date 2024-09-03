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

import enum
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class ReservedKeys(str, enum.Enum):
    """Contains constants used to identify all featurestore keys that
    may not be to used by users. Avoids overwriting system data"""

    MLI_NOTIFY_CONSUMERS = "_SMARTSIM_MLI_NOTIFY_CONSUMERS"
    """Storage location for the list of registered consumers that will receive
    events from an EventBroadcaster"""

    @classmethod
    def contains(cls, value: str) -> bool:
        """Convert a string representation into an enumeration member

        :param value: the string to convert
        :returns: the enumeration member if the conversion succeeded, otherwise None"""
        try:
            cls(value)
        except ValueError:
            return False

        return True


@dataclass(frozen=True)
class FeatureStoreKey:
    """A key,descriptor pair enabling retrieval of an item from a feature store"""

    key: str
    """The unique key of an item in a feature store"""
    descriptor: str
    """The unique identifier of the feature store containing the key"""

    def __post_init__(self) -> None:
        """Ensure the key and descriptor have at least one character

        :raises ValueError: if key or descriptor are empty strings
        """
        if len(self.key) < 1:
            raise ValueError("Key must have at least one character.")
        if len(self.descriptor) < 1:
            raise ValueError("Descriptor must have at least one character.")


class FeatureStore(ABC):
    """Abstract base class providing the common interface for retrieving
    values from a feature store implementation"""

    def __init__(self, descriptor: str, allow_reserved_writes: bool = False) -> None:
        """Initialize the feature store

        :param descriptor: the stringified version of a storage descriptor
        :param allow_reserved_writes: override the default behavior of blocking
        writes to reserved keys"""
        self._enable_reserved_writes = allow_reserved_writes
        """Flag used to ensure that any keys written by the system to a feature store
        are not overwritten by user code. Disabled by default. Subclasses must set the
        value intentionally."""
        self._descriptor = descriptor
        """Stringified version of the unique ID enabling a client to connect
        to the feature store"""

    def _check_reserved(self, key: str) -> None:
        """A utility method used to verify access to write to a reserved key
        in the FeatureStore. Used by subclasses in __setitem___ implementations

        :param key: a key to compare to the reserved keys
        :raises SmartSimError: if the key is reserved"""
        if not self._enable_reserved_writes and ReservedKeys.contains(key):
            raise SmartSimError(
                "Use of reserved key denied. "
                "Unable to overwrite system configuration"
            )

    def __getitem__(self, key: str) -> t.Union[str, bytes]:
        """Retrieve an item using key

        :param key: Unique key of an item to retrieve from the feature store"""
        try:
            return self._get(key)
        except KeyError as ex:
            raise SmartSimError(f"An unknown key was requested: {key}") from ex
        except Exception as ex:
            # note: explicitly avoid round-trip to check for key existence
            raise SmartSimError(
                f"Could not get value for existing key {key}, error:\n{ex}"
            ) from ex

    def __setitem__(self, key: str, value: t.Union[str, bytes]) -> None:
        """Assign a value using key

        :param key: Unique key of an item to set in the feature store
        :param value: Value to persist in the feature store"""
        self._check_reserved(key)
        self._set(key, value)

    def __contains__(self, key: str) -> bool:
        """Membership operator to test for a key existing within the feature store.

        :param key: Unique key of an item to retrieve from the feature store
        :returns: `True` if the key is found, `False` otherwise"""
        return self._contains(key)

    @abstractmethod
    def _get(self, key: str) -> t.Union[str, bytes]:
        """Retrieve a value from the underlying stroage mechanism

        :param key: The unique key that identifies the resource
        :returns: the value identified by the key
        :raises KeyError: if the key has not been used to store a value"""

    @abstractmethod
    def _set(self, key: str, value: t.Union[str, bytes]) -> None:
        """Store a value into the underlying stroage mechanism

        :param key: The unique key that identifies the resource
        :param value: The value to store
        :returns: the value identified by the key
        :raises KeyError: if the key has not been used to store a value"""

    @abstractmethod
    def _contains(self, key: str) -> bool:
        """Determine if the storage mechanism contains a given key

        :param key: The unique key that identifies the resource
        :returns: `True` if the key is defined, `False` otherwise"""

    @property
    def _allow_reserved_writes(self) -> bool:
        """Return the boolean flag indicating if writing to reserved keys is
        enabled for this feature store

        :returns: `True` if enabled, `False` otherwise"""
        return self._enable_reserved_writes

    @_allow_reserved_writes.setter
    def _allow_reserved_writes(self, value: bool) -> None:
        """Modify the boolean flag indicating if writing to reserved keys is
        enabled for this feature store

        :param value: the new value to set for the flag"""
        self._enable_reserved_writes = value

    @property
    def descriptor(self) -> str:
        """Unique identifier enabling a client to connect to the feature store

        :returns: A descriptor encoded as a string"""
        return self._descriptor
