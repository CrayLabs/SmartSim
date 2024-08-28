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
import sys
import typing as t

import smartsim._core.mli.comm.channel.channel as cch
from smartsim.log import get_logger

logger = get_logger(__name__)

import dragon.channels as dch


class DragonCommChannel(cch.CommChannelBase):
    """Passes messages by writing to a Dragon channel"""

    def __init__(self, channel: "dch.Channel", recv_timeout: int = 0) -> None:
        """Initialize the DragonCommChannel instance

        :param channel: a channel to use for communications
        :param recv_timeout: a default timeout to apply to receive calls"""
        serialized_ch = channel.serialize()
        safe_descriptor = base64.b64encode(serialized_ch).decode("utf-8")
        super().__init__(safe_descriptor)
        self._channel = channel
        self._recv_timeout = recv_timeout

    @property
    def recv_timeout(self) -> int:
        """The timeout for receive requests (in seconds)"""
        return self._recv_timeout

    def send(self, value: bytes) -> None:
        """Send a message throuh the underlying communication channel
        :param value: The value to send"""
        with self._channel.sendh(timeout=None) as sendh:
            sendh.send_bytes(value)

    def recv(self, timeout: int = 0) -> t.List[bytes]:
        """Receives message(s) through the underlying communication channel

        :param timeout: maximum time to wait for messages to arrive
        :returns: the received message"""
        with self._channel.recvh(timeout=timeout) as recvh:
            messages: t.List[bytes] = []

            # todo: consider that this could (under load) never exit. do we need
            # to configure a maximum number to pull at once?
            try:
                timeout = timeout or self._recv_timeout
                while message_bytes := recvh.recv_bytes(timeout=timeout):
                    messages.append(message_bytes)
            except dch.ChannelEmpty:
                ...  # emptied the queue, swallow this ex

            return messages

    @property
    def descriptor_string(self) -> str:
        """Return the channel descriptor for the underlying dragon channel
        as a string. Automatically performs base64 encoding to ensure the
        string can be used in a call to `from_descriptor`"""
        if isinstance(self._descriptor, str):
            return self._descriptor

        if isinstance(self._descriptor, bytes):
            return base64.b64encode(self._descriptor).decode("utf-8")

        raise ValueError(f"Unable to convert channel descriptor: {self._descriptor}")

    @classmethod
    def from_descriptor(
        cls,
        descriptor: str,
    ) -> "DragonCommChannel":
        """A factory method that creates an instance from a descriptor string

        :param descriptor: The descriptor that uniquely identifies the resource. Output
        from `descriptor_string` is correctly encoded.
        :returns: An attached DragonCommChannel"""
        try:
            utf8_descriptor = descriptor.encode("utf-8")
            actual_descriptor = base64.b64decode(utf8_descriptor)
            channel = dch.Channel.attach(actual_descriptor)
            return DragonCommChannel(channel)
        except:
            logger.error(
                f"Failed to create dragon comm channel: {descriptor}", exc_info=True
            )
            raise
