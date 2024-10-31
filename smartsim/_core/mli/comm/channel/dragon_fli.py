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

# isort: off

import dragon
import dragon.fli as fli
from dragon.channels import Channel

# isort: on

import typing as t

import smartsim._core.mli.comm.channel.channel as cch
import smartsim._core.mli.comm.channel.dragon_util as drg_util
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class DragonFLIChannel(cch.CommChannelBase):
    """Passes messages by writing to a Dragon FLI Channel."""

    def __init__(
        self,
        fli_: fli.FLInterface,
        buffer_size: int = drg_util.DEFAULT_CHANNEL_BUFFER_SIZE,
    ) -> None:
        """Initialize the DragonFLIChannel instance.

        :param fli_: The FLIInterface to use as the underlying communications channel
        :param sender_supplied: Flag indicating if the FLI uses sender-supplied streams
        :param buffer_size: Maximum number of sent messages that can be buffered
        """
        descriptor = drg_util.channel_to_descriptor(fli_)
        super().__init__(descriptor)

        self._channel: t.Optional["Channel"] = None
        """The underlying dragon Channel used by a sender-side DragonFLIChannel
        to attach to the main FLI channel"""

        self._fli = fli_
        """The underlying dragon FLInterface used by this CommChannel for communications"""
        self._buffer_size: int = buffer_size
        """Maximum number of messages that can be buffered before sending"""

    def send(self, value: bytes, timeout: float = 0.001) -> None:
        """Send a message through the underlying communication channel.

        :param value: The value to send
        :param timeout: Maximum time to wait (in seconds) for messages to send
        :raises SmartSimError: If sending message fails
        """
        try:
            if self._channel is None:
                self._channel = drg_util.create_local(self._buffer_size)

            with self._fli.sendh(timeout=None, stream_channel=self._channel) as sendh:
                sendh.send_bytes(value, timeout=timeout)
                logger.debug(f"DragonFLIChannel {self.descriptor} sent message")
        except Exception as e:
            self._channel = None
            raise SmartSimError(
                f"Error sending via DragonFLIChannel {self.descriptor}"
            ) from e

    def send_multiple(
        self,
        values: t.Sequence[bytes],
        timeout: float = 0.001,
    ) -> None:
        """Send a message through the underlying communication channel.

        :param values: The values to send
        :param timeout: Maximum time to wait (in seconds) for messages to send
        :raises SmartSimError: If sending message fails
        """
        try:
            if self._channel is None:
                self._channel = drg_util.create_local(self._buffer_size)

            with self._fli.sendh(timeout=None, stream_channel=self._channel) as sendh:
                for value in values:
                    sendh.send_bytes(value)
                    logger.debug(f"DragonFLIChannel {self.descriptor} sent message")
        except Exception as e:
            self._channel = None
            raise SmartSimError(
                f"Error sending via DragonFLIChannel {self.descriptor} {e}"
            ) from e

    def recv(self, timeout: float = 0.001) -> t.List[bytes]:
        """Receives message(s) through the underlying communication channel.

        :param timeout: Maximum time to wait (in seconds) for messages to arrive
        :returns: The received message(s)
        :raises SmartSimError: If receiving message(s) fails
        """
        messages = []
        eot = False
        with self._fli.recvh(timeout=timeout) as recvh:
            while not eot:
                try:
                    message, _ = recvh.recv_bytes(timeout=timeout)
                    messages.append(message)
                    logger.debug(f"DragonFLIChannel {self.descriptor} received message")
                except fli.FLIEOT:
                    eot = True
                    logger.debug(f"DragonFLIChannel exhausted: {self.descriptor}")
                except Exception as e:
                    raise SmartSimError(
                        f"Error receiving messages: DragonFLIChannel {self.descriptor}"
                    ) from e
        return messages

    @classmethod
    def from_descriptor(
        cls,
        descriptor: str,
    ) -> "DragonFLIChannel":
        """A factory method that creates an instance from a descriptor string.

        :param descriptor: The descriptor that uniquely identifies the resource
        :returns: An attached DragonFLIChannel
        :raises SmartSimError: If creation of DragonFLIChannel fails
        :raises ValueError: If the descriptor is invalid
        """
        if not descriptor:
            raise ValueError("Invalid descriptor provided")

        try:
            return DragonFLIChannel(fli_=drg_util.descriptor_to_fli(descriptor))
        except Exception as e:
            raise SmartSimError(
                f"Error while creating DragonFLIChannel: {descriptor}"
            ) from e
