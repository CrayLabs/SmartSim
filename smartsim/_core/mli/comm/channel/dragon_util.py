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
import binascii
import typing as t

import dragon.channels as dch
import dragon.fli as fli
import dragon.infrastructure.facts as df
import dragon.infrastructure.parameters as dp
import dragon.managed_memory as dm
import dragon.utils as du

from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)

DEFAULT_CHANNEL_BUFFER_SIZE = 500
"""Maximum number of messages that can be buffered. DragonCommChannel will
raise an exception if no clients consume messages before the buffer is filled."""

LAST_OFFSET = 0
"""The last offset used to create a local channel. This is used to avoid
unnecessary retries when creating a local channel."""


def channel_to_descriptor(channel: t.Union[dch.Channel, fli.FLInterface]) -> str:
    """Convert a dragon channel to a descriptor string.

    :param channel: The dragon channel to convert
    :returns: The descriptor string
    :raises: SmartSimError if a dragon channel is not provided
    """
    if channel is None:
        raise SmartSimError("Channel is not available to create a descriptor")

    serialized_ch = channel.serialize()
    return base64.b64encode(serialized_ch).decode("utf-8")


def pool_to_descriptor(pool: dm.MemoryPool) -> str:
    """Convert a dragon memory pool to a descriptor string.

    :param pool: The memory pool to convert
    :returns: The descriptor string"""
    if pool is None:
        raise SmartSimError("Memory pool is not available to create a descriptor")

    serialized_pool = pool.serialize()
    return base64.b64encode(serialized_pool).decode("utf-8")


def descriptor_to_fli(descriptor: str) -> "fli.FLInterface":
    """Create and attach a new FLI instance given
    the string-encoded descriptor.

    :param descriptor: The descriptor of an FLI to attach to
    :returns: The attached dragon FLI
    :raises ValueError: If the descriptor is empty or incorrectly formatted
    """
    if len(descriptor) < 1:
        raise ValueError("Descriptors may not be empty")

    try:
        encoded = descriptor.encode("utf-8")
        descriptor_ = base64.b64decode(encoded)
        return fli.FLInterface.attach(descriptor_)
    except binascii.Error:
        raise ValueError("The descriptor was not properly base64 encoded")
    except fli.DragonFLIError:
        raise SmartSimError("The descriptor did not address an available FLI")


def descriptor_to_channel(descriptor: str) -> dch.Channel:
    """Create and attach a new Channel instance given
    the string-encoded descriptor.

    :param descriptor: The descriptor of a channel to attach to
    :returns: The attached dragon Channel
    :raises ValueError: If the descriptor is empty or incorrectly formatted
    :raises SmartSimError: If the descriptor does not attach to a channel"""
    if len(descriptor) < 1:
        raise ValueError("Descriptors may not be empty")

    try:
        encoded = descriptor.encode("utf-8")
        descriptor_ = base64.b64decode(encoded)
        return dch.Channel.attach(descriptor_)
    except binascii.Error:
        raise ValueError("The descriptor was not properly base64 encoded")
    except dch.ChannelError:
        raise SmartSimError("The descriptor did not address an available channel")


def create_local(_capacity: int = 0) -> dch.Channel:
    """Creates a Channel attached to the local memory pool. Replacement for
    direct calls to `dch.Channel.make_process_local()` to enable
    supplying a channel capacity.

    :param capacity: The number of events the channel can buffer; uses the default
    buffer size `DEFAULT_CHANNEL_BUFFER_SIZE` when not supplied
    :returns: The instantiated channel
    :raises SmartSimError: If unable to attach local channel
    """
    # current implementation has a bug wrt MPI that must be fixed.
    # falling back to `make_process_local` and disabling buffer size tests

    # pool = dm.MemoryPool.attach(du.B64.str_to_bytes(dp.this_process.default_pd))
    # pool_descriptor = pool_to_descriptor(pool)
    # channel: t.Optional[dch.Channel] = None
    # offset = 0

    # global LAST_OFFSET
    # if LAST_OFFSET:
    #     offset = LAST_OFFSET

    # capacity = capacity if capacity > 0 else DEFAULT_CHANNEL_BUFFER_SIZE

    # while not channel:
    #     # search for an open channel ID
    #     offset += 1
    #     channel_id = df.BASE_USER_MANAGED_CUID + offset
    #     try:
    #         channel = dch.Channel(mem_pool=pool, c_uid=channel_id, capacity=capacity)
    #         LAST_OFFSET = offset
    #         descriptor = channel_to_descriptor(channel)
    #         logger.debug(
    #             "Local channel created: "
    #             f"{channel_id=}, {pool_descriptor=}, {capacity=}, {descriptor=}"
    #         )
    #     except dch.ChannelError as e:
    #         if offset < 100:
    #             logger.warning(f"Channnel id `{channel_id}` is not open. Retrying...")
    #         else:
    #             LAST_OFFSET = 0
    #             logger.error(f"All attempts to attach local channel have failed")
    #             raise SmartSimError("Failed to attach local channel") from e
    channel = dch.Channel.make_process_local()
    return channel
