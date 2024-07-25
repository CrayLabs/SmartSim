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

from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore
from smartsim.error import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class DragonFeatureStore(FeatureStore):
    """A feature store backed by a dragon distributed dictionary"""

    def __init__(self, storage: "dragon_ddict.DDict") -> None:
        """Initialize the DragonFeatureStore instance"""
        self._storage = storage

    def __getitem__(self, key: str) -> t.Union[str, bytes]:
        """Retrieve an item using key
        :param key: Unique key of an item to retrieve from the feature store"""
        try:
            value: t.Union[str, bytes] = self._storage[key]
            return value
        except KeyError as ex:
            raise ex
        except Exception as ex:
            # note: explicitly avoid round-trip to check for key existence
            raise SmartSimError(
                f"Could not get value for existing key {key}, error:\n{ex}"
            ) from ex

    def __setitem__(self, key: str, value: t.Union[str, bytes]) -> None:
        """Assign a value using key
        :param key: Unique key of an item to set in the feature store
        :param value: Value to persist in the feature store"""
        self._storage[key] = value

    def __contains__(self, key: str) -> bool:
        """Membership operator to test for a key existing within the feature store.
        Return `True` if the key is found, `False` otherwise
        :param key: Unique key of an item to retrieve from the feature store"""
        return key in self._storage

    @property
    def descriptor(self) -> str:
        """Return a unique identifier enabling a client to connect to
        the feature store
        :returns: A descriptor encoded as a string"""
        return str(self._storage.serialize())

    @classmethod
    def from_descriptor(
        cls,
        descriptor: str,
        # b64encoded: bool = False,
    ) -> "DragonFeatureStore":
        # import dragon.data.ddict.ddict as dragon_ddict  # pylint: disable=import-outside-toplevel

        # # if b64encoded:
        # #     descriptor = base64.b64decode(descriptor).encode("utf-8")
        # # ddict = DDict.attach(descriptor)
        # # ddict.attach(descriptor)

        # storage = dragon_ddict.DDict()
        # storage.attach(descriptor)
        # return DragonFeatureStore(storage)

        if descriptor is None:
            print("foo")
            return None
        return DragonFeatureStore({"tmp": "here"})