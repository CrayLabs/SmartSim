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
from dragon import fli
import dragon.channels as dch
import dragon.infrastructure.facts as df
import dragon.infrastructure.parameters as dp
import dragon.managed_memory as dm
import dragon.utils as du

# isort: on

import base64
import typing as t

import smartsim._core.mli.comm.channel.channel as cch
from smartsim._core.mli.comm.channel.dragon_channel import create_local
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class DragonFLIChannel(cch.CommChannelBase):
    """Passes messages by writing to a Dragon FLI Channel."""

    def __init__(
        self,
        fli_: fli.FLInterface,
        sender_supplied: bool = True,
        buffer_size: int = 0,
    ) -> None:
        """Initialize the DragonFLIChannel instance.

        :param fli_desc: The descriptor of the FLI channel to attach
        :param sender_supplied: Flag indicating if the FLI uses sender-supplied streams
        :param buffer_size: Maximum number of sent messages that can be buffered
        """
        descriptor = base64.b64encode(fli_.serialize()).decode("utf-8")
        super().__init__(descriptor)

        self._fli = fli_
        self._channel: t.Optional["dch.Channel"] = (
            create_local(buffer_size) if sender_supplied else None
        )

    def send(
        self, value: bytes, timeout: float = 0.001, blocking: bool = False
    ) -> None:
        """Send a message through the underlying communication channel.

        :param value: The value to send
        :param timeout: Maximum time to wait (in seconds) for messages to send
        :param blocking: Block returning until the message has been received
        :raises SmartSimError: If sending message fails
        """
        try:
            with self._fli.sendh(timeout=None, stream_channel=self._channel) as sendh:
                sendh.send_bytes(value, timeout=timeout)
                logger.debug(f"DragonFLIChannel {self.descriptor} sent message")
        except Exception as e:
            raise SmartSimError(
                f"Error sending message: DragonFLIChannel {self.descriptor}"
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
                except Exception as e:
                    raise SmartSimError(
                        f"Error receiving messages: DragonFLIChannel {self.descriptor}"
                    ) from e
        return messages

    @classmethod
    def _string_descriptor_to_fli(cls, descriptor: str) -> "fli.FLInterface":
        """Helper method to convert a string-safe, encoded descriptor back
        into its original byte format"""
        descriptor_ = base64.b64decode(descriptor.encode("utf-8"))
        return fli.FLInterface.attach(descriptor_)

    @classmethod
    def from_sender_supplied_descriptor(
        cls,
        descriptor: str,
    ) -> "DragonFLIChannel":
        """A factory method that creates an instance from a descriptor string

        :param descriptor: the descriptor of the main FLI channel to attach
        :returns: An attached DragonFLIChannel"""
        try:
            return DragonFLIChannel(
                fli_=cls._string_descriptor_to_fli(descriptor),
                sender_supplied=True,
            )
        except:
            logger.error(
                f"Error while creating sender supplied DragonFLIChannel: {descriptor}"
            )
            raise

    @classmethod
    def from_descriptor(
        cls,
        descriptor: str,
    ) -> "DragonFLIChannel":
        """A factory method that creates an instance from a descriptor string.

        :param descriptor: The descriptor that uniquely identifies the resource
        :returns: An attached DragonFLIChannel
        :raises SmartSimError: If creation of DragonFLIChanenel fails
        """
        if not descriptor:
            raise ValueError("Invalid descriptor provided")

        try:
            return DragonFLIChannel(
                fli_=cls._string_descriptor_to_fli(descriptor),
                sender_supplied=False,
            )
        except Exception as e:
            raise SmartSimError(
                f"Error while creating DragonFLIChannel: {descriptor}"
            ) from e
