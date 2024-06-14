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

import smartsim.error as sse
from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore
from smartsim.log import get_logger

if t.TYPE_CHECKING:
    from dragon.data.distdictionary.dragon_dict import DragonDict


logger = get_logger(__name__)


class DragonFeatureStore(FeatureStore):
    """A feature store backed by a dragon distributed dictionary"""

    def __init__(self, storage: "DragonDict") -> None:
        """Initialize the DragonFeatureStore instance"""
        self._storage = storage

    def __getitem__(self, key: str) -> t.Any:
        """Retrieve an item using key
        :param key: Unique key of an item to retrieve from the feature store"""
        key_ = key.encode("utf-8")
        if key_ not in self._storage:
            raise sse.SmartSimError(f"{key} not found in feature store")
        return self._storage[key_]

    def __setitem__(self, key: str, value: bytes) -> None:
        """Assign a value using key
        :param key: Unique key of an item to set in the feature store
        :param value: Value to persist in the feature store"""
        key_ = key.encode("utf-8")
        self._storage[key_] = value

    def __contains__(self, key: t.Union[str, bytes]) -> bool:
        """Membership operator to test for a key existing within the feature store.
        Return `True` if the key is found, `False` otherwise
        :param key: Unique key of an item to retrieve from the feature store"""
        if isinstance(key, str):
            key = key.encode("utf-8")
        return key in self._storage
