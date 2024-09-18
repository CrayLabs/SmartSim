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
import typing as t
import uuid
from abc import ABC, abstractmethod

from smartsim.log import get_logger

logger = get_logger(__name__)


class CommChannelBase(ABC):
    """Base class for abstracting a message passing mechanism"""

    def __init__(
        self,
        descriptor: str,
        name: t.Optional[str] = None,
    ) -> None:
        """Initialize the CommChannel instance.

        :param descriptor: Channel descriptor
        """
        self._descriptor = descriptor
        """An opaque identifier used to connect to an underlying communication channel"""
        self._name = name or str(uuid.uuid4())
        """A user-friendly identifier for channel-related logging"""

    @abstractmethod
    def send(self, value: bytes, timeout: float = 0) -> None:
        """Send a message through the underlying communication channel.

        :param timeout: Maximum time to wait (in seconds) for messages to send
        :param value: The value to send
        :raises SmartSimError: If sending message fails
        """

    @abstractmethod
    def recv(self, timeout: float = 0) -> t.List[bytes]:
        """Receives message(s) through the underlying communication channel.

        :param timeout: Maximum time to wait (in seconds) for messages to arrive
        :returns: The received message
        """

    @property
    def descriptor(self) -> str:
        """Return the channel descriptor for the underlying dragon channel.

        :returns: Byte encoded channel descriptor
        """
        return self._descriptor

    @property
    def decoded_descriptor(self) -> bytes:
        """Return the descriptor decoded from a string into bytes"""
        return base64.b64decode(self._descriptor.encode("utf-8"))

    def __str__(self) -> str:
        """Build a string representation of the channel useful for printing"""
        classname = type(self).__class__.__name__
        return f"{classname}('{self._name}', '{self._descriptor}')"
