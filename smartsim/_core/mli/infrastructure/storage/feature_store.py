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
    def from_string(cls, value: str) -> t.Optional["ReservedKeys"]:
        """Convert a string representation into an enumeration member

        :param value: the string to convert
        :returns: the enumeration member if the conversion succeeded, otherwise None"""
        try:
            return cls(value)
        except ValueError:
            ...  # value is not reserved, swallow

        return None


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

    def __init__(self) -> None:
        self._reserved_write_enabled = False

    def _check_reserved(self, key: str) -> None:
        """A utility method used to verify access to write to a reserved key
        in the FeatureStore. Used by subclasses in __setitem___ implementations

        :param key: a key to compare to the reserved keys
        :raises SmartSimError: if the key is reserved"""
        reserved_key_match = ReservedKeys.from_string(key)
        if reserved_key_match and not self._reserved_write_enabled:
            raise SmartSimError(
                "Use of reserved key denied. "
                "Unable to overwrite system configuration"
            )

    @abstractmethod
    def __getitem__(self, key: str) -> t.Union[str, bytes]:
        """Retrieve an item using key

        :param key: Unique key of an item to retrieve from the feature store"""

    @abstractmethod
    def __setitem__(self, key: str, value: t.Union[str, bytes]) -> None:
        """Assign a value using key

        :param key: Unique key of an item to set in the feature store
        :param value: Value to persist in the feature store"""

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """Membership operator to test for a key existing within the feature store.

        :param key: Unique key of an item to retrieve from the feature store
        :returns: `True` if the key is found, `False` otherwise"""

    @property
    @abstractmethod
    def descriptor(self) -> str:
        """Unique identifier enabling a client to connect to the feature store

        :returns: A descriptor encoded as a string"""
