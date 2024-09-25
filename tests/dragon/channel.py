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

import base64
import pathlib
import threading
import typing as t

from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class FileSystemCommChannel(CommChannelBase):
    """Passes messages by writing to a file"""

    def __init__(self, key: pathlib.Path) -> None:
        """Initialize the FileSystemCommChannel instance.

        :param key: a path to the root directory of the feature store
        """
        self._lock = threading.RLock()

        super().__init__(key.as_posix())
        self._file_path = key

        if not self._file_path.parent.exists():
            self._file_path.parent.mkdir(parents=True)

        self._file_path.touch()

    def send(self, value: bytes, timeout: float = 0) -> None:
        """Send a message throuh the underlying communication channel.

        :param value: The value to send
        :param timeout: maximum time to wait (in seconds) for messages to send
        """
        with self._lock:
            # write as text so we can add newlines as delimiters
            with open(self._file_path, "a") as fp:
                encoded_value = base64.b64encode(value).decode("utf-8")
                fp.write(f"{encoded_value}\n")
                logger.debug(f"FileSystemCommChannel {self._file_path} sent message")

    def recv(self, timeout: float = 0) -> t.List[bytes]:
        """Receives message(s) through the underlying communication channel.

        :param timeout: maximum time to wait (in seconds) for messages to arrive
        :returns: the received message
        :raises SmartSimError: if the descriptor points to a missing file
        """
        with self._lock:
            messages: t.List[bytes] = []
            if not self._file_path.exists():
                raise SmartSimError("Empty channel")

            # read as text so we can split on newlines
            with open(self._file_path, "r") as fp:
                lines = fp.readlines()

            if lines:
                line = lines.pop(0)
                event_bytes = base64.b64decode(line.encode("utf-8"))
                messages.append(event_bytes)

            self.clear()

            # remove the first message only, write remainder back...
            if len(lines) > 0:
                with open(self._file_path, "w") as fp:
                    fp.writelines(lines)

                logger.debug(
                    f"FileSystemCommChannel {self._file_path} received message"
                )

            return messages

    def clear(self) -> None:
        """Create an empty file for events."""
        if self._file_path.exists():
            self._file_path.unlink()
        self._file_path.touch()

    @classmethod
    def from_descriptor(
        cls,
        descriptor: str,
    ) -> "FileSystemCommChannel":
        """A factory method that creates an instance from a descriptor string.

        :param descriptor: The descriptor that uniquely identifies the resource
        :returns: An attached FileSystemCommChannel
        """
        try:
            path = pathlib.Path(descriptor)
            return FileSystemCommChannel(path)
        except:
            logger.warning(f"failed to create fs comm channel: {descriptor}")
            raise
