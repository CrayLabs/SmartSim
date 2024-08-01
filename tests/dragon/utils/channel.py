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
import threading
import typing as t

from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim.log import get_logger

logger = get_logger(__name__)


class FileSystemCommChannel(CommChannelBase):
    """Passes messages by writing to a file"""

    def __init__(self, key: t.Union[bytes, pathlib.Path]) -> None:
        """Initialize the FileSystemCommChannel instance
        :param key: a path to the root directory of the feature store"""
        self._lock = threading.RLock()

        if not isinstance(key, bytes):
            super().__init__(key.as_posix().encode("utf-8"))
            self._file_path = key
        else:
            super().__init__(key)
            self._file_path = pathlib.Path(key.decode("utf-8"))

        if not self._file_path.parent.exists():
            self._file_path.parent.mkdir(parents=True)

        self._file_path.touch()

    def send(self, value: bytes) -> None:
        """Send a message throuh the underlying communication channel
        :param value: The value to send"""
        logger.debug(
            f"Channel {self.descriptor.decode('utf-8')} sending message to {self._file_path}"
        )
        with self._lock:
            self._file_path.write_bytes(value)

    def recv(self) -> bytes:
        """Receieve a message through the underlying communication channel
        :returns: the received message"""
        with self._lock:
            if self._file_path.exists():
                incoming = self._file_path.read_bytes()
                self._file_path.unlink()
                return incoming

    @classmethod
    def from_descriptor(
        cls,
        descriptor: t.Union[str, bytes],
    ) -> "FileSystemCommChannel":
        """A factory method that creates an instance from a descriptor string
        :param descriptor: The descriptor that uniquely identifies the resource
        :returns: An attached FileSystemCommChannel"""
        try:
            if isinstance(descriptor, str):
                path = pathlib.Path(descriptor)
            else:
                path = pathlib.Path(descriptor.decode("utf-8"))
            return FileSystemCommChannel(path)
        except:
            print("failed to create FS comm channel: {descriptor}")
