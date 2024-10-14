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

import dragon.channels as dch

import smartsim._core.mli.comm.channel.channel as cch
import smartsim._core.mli.comm.channel.dragon_util as drg_util
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class DragonCommChannel(cch.CommChannelBase):
    """Passes messages by writing to a Dragon channel."""

    def __init__(self, channel: "dch.Channel") -> None:
        """Initialize the DragonCommChannel instance.

        :param channel: A channel to use for communications
        """
        descriptor = drg_util.channel_to_descriptor(channel)
        super().__init__(descriptor)
        self._channel = channel
        """The underlying dragon channel used by this CommChannel for communications"""

    @property
    def channel(self) -> "dch.Channel":
        """The underlying communication channel.

        :returns: The channel
        """
        return self._channel

    def send(
        self,
        value: bytes,
        timeout: t.Optional[float] = 0.001,
        handle_timeout: float = 0.001,
    ) -> None:
        """Send a message through the underlying communication channel.

        :param value: The value to send
        :param timeout: Maximum time to wait (in seconds) for messages to be sent
        :param handle_timeout: Maximum time to wait to obtain new send handle
        :raises SmartSimError: If sending message fails
        """
        try:
            with self._channel.sendh(timeout=handle_timeout) as sendh:
                sendh.send_bytes(value, timeout=timeout)
                logger.debug(f"DragonCommChannel {self.descriptor} sent message")
        except Exception as e:
            raise SmartSimError(
                f"Error sending via DragonCommChannel {self.descriptor}"
            ) from e

    def recv(
        self, timeout: t.Optional[float] = 0.001, handle_timeout: float = 0.001
    ) -> t.List[bytes]:
        """Receives message(s) through the underlying communication channel.

        :param timeout: Maximum time to wait (in seconds) for message to arrive
        :param handle_timeout: Maximum time to wait to obtain new receive handle
        :returns: The received message(s)
        """
        with self._channel.recvh(timeout=handle_timeout) as recvh:
            messages: t.List[bytes] = []

            try:
                message_bytes = recvh.recv_bytes(timeout=timeout)
                messages.append(message_bytes)
                logger.debug(f"DragonCommChannel {self.descriptor} received message")
            except dch.ChannelEmpty:
                # emptied the queue, ok to swallow this ex
                logger.debug(f"DragonCommChannel exhausted: {self.descriptor}")
            except dch.ChannelRecvTimeout:
                logger.debug(f"Timeout exceeded on channel.recv: {self.descriptor}")

            return messages

    @classmethod
    def from_descriptor(
        cls,
        descriptor: str,
    ) -> "DragonCommChannel":
        """A factory method that creates an instance from a descriptor string.

        :param descriptor: The descriptor that uniquely identifies the resource.
        :returns: An attached DragonCommChannel
        :raises SmartSimError: If creation of comm channel fails
        """
        try:
            channel = drg_util.descriptor_to_channel(descriptor)
            return DragonCommChannel(channel)
        except Exception as ex:
            raise SmartSimError(
                f"Failed to create dragon comm channel: {descriptor}"
            ) from ex

    @classmethod
    def from_local(cls, _descriptor: t.Optional[str] = None) -> "DragonCommChannel":
        """A factory method that creates a local channel instance.

        :param _descriptor: Unused placeholder
        :returns: An attached DragonCommChannel"""
        try:
            channel = drg_util.create_local()
            return DragonCommChannel(channel)
        except:
            logger.error(f"Failed to create local dragon comm channel", exc_info=True)
            raise
