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
from abc import ABC, abstractmethod

from smartsim.log import get_logger

logger = get_logger(__name__)


class CommChannelBase(ABC):
    """Base class for abstracting a message passing mechanism"""

    def __init__(self, descriptor: t.Union[str, bytes]) -> None:
        """Initialize the CommChannel instance"""
        self._descriptor = descriptor

    @abstractmethod
    def send(self, value: bytes) -> None:
        """Send a message through the underlying communication channel

        :param value: The value to send"""

    @abstractmethod
    def recv(self, timeout: int = 0) -> t.List[bytes]:
        """Receives message(s) through the underlying communication channel

        :param timeout: maximum time to wait for messages to arrive
        :returns: the received message"""

    @property
    def descriptor(self) -> bytes:
        """Return the channel descriptor for the underlying dragon channel"""
        if isinstance(self._descriptor, str):
            return self._descriptor.encode("utf-8")
        return self._descriptor
