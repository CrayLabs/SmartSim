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

import pathlib
import typing as t

import smartsim.error as sse
from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore


class MemoryFeatureStore(FeatureStore):
    """A feature store with values persisted only in local memory"""

    def __init__(self) -> None:
        """Initialize the MemoryFeatureStore instance"""
        self._storage: t.Dict[str, bytes] = {}

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item using key
        :param key: Unique key of an item to retrieve from the feature store"""
        if key not in self._storage:
            raise sse.SmartSimError(f"{key} not found in feature store")
        return self._storage[key]

    def __setitem__(self, key: str, value: bytes) -> None:
        """Membership operator to test for a key existing within the feature store.
        Return `True` if the key is found, `False` otherwise
        :param key: Unique key of an item to retrieve from the feature store"""
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
        return "in-memory-fs"


class FileSystemFeatureStore(FeatureStore):
    """Alternative feature store implementation for testing. Stores all
    data on the file system"""

    def __init__(self, storage_dir: t.Optional[pathlib.Path] = None) -> None:
        """Initialize the FileSystemFeatureStore instance
        :param storage_dir: (optional) root directory to store all data relative to"""
        self._storage_dir = storage_dir

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item using key
        :param key: Unique key of an item to retrieve from the feature store"""
        path = self._key_path(key)
        if not path.exists():
            raise sse.SmartSimError(f"{path} not found in feature store")
        return path.read_bytes()

    def __setitem__(self, key: str, value: bytes) -> None:
        """Assign a value using key
        :param key: Unique key of an item to set in the feature store
        :param value: Value to persist in the feature store"""
        path = self._key_path(key, create=True)
        path.write_bytes(value)

    def __contains__(self, key: str) -> bool:
        """Membership operator to test for a key existing within the feature store.
        Return `True` if the key is found, `False` otherwise
        :param key: Unique key of an item to retrieve from the feature store"""
        path = self._key_path(key)
        return path.exists()

    def _key_path(self, key: str, create: bool = False) -> pathlib.Path:
        """Given a key, return a path that is optionally combined with a base
        directory used by the FileSystemFeatureStore.
        :param key: Unique key of an item to retrieve from the feature store"""
        value = pathlib.Path(key)

        if self._storage_dir:
            value = self._storage_dir / key

        if create:
            value.parent.mkdir(parents=True, exist_ok=True)

        return value

    @property
    def descriptor(self) -> str:
        """Return a unique identifier enabling a client to connect to
        the feature store
        :returns: A descriptor encoded as a string"""
        if not self._storage_dir:
            raise ValueError("No storage path configured")
        return self._storage_dir.as_posix()

    @classmethod
    def from_descriptor(
        cls,
        descriptor: str,
        # b64encoded: bool = False,
    ) -> "FileSystemFeatureStore":
        # if b64encoded:
        #     descriptor = base64.b64decode(descriptor).encode("utf-8")
        path = pathlib.Path(descriptor)
        if not path.is_dir():
            raise ValueError("FileSystemFeatureStore requires a directory path")
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return FileSystemFeatureStore(path)


class DragonDict:
    """Mock implementation of a dragon dictionary"""

    def __init__(self) -> None:
        """Initialize the mock DragonDict instance"""
        self._storage: t.Dict[bytes, t.Any] = {}

    def __getitem__(self, key: bytes) -> t.Any:
        """Retrieve an item using key
        :param key: Unique key of an item to retrieve from the feature store"""
        return self._storage[key]

    def __setitem__(self, key: bytes, value: t.Any) -> None:
        """Assign a value using key
        :param key: Unique key of an item to set in the feature store
        :param value: Value to persist in the feature store"""
        self._storage[key] = value

    def __contains__(self, key: bytes) -> bool:
        """Return `True` if the key is found, `False` otherwise
        :param key: Unique key of an item to retrieve from the feature store"""
        return key in self._storage
