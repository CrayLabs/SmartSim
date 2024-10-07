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
import uuid
from collections import defaultdict, deque

from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim._core.mli.infrastructure.comm.event import EventBase
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class BroadcastResult(t.NamedTuple):
    """Contains summary details about a broadcast."""

    num_sent: int
    """The total number of messages delivered across all consumers"""
    num_failed: int
    """The total number of messages not delivered across all consumers"""


class EventBroadcaster:
    """Performs fan-out publishing of system events."""

    def __init__(
        self,
        backbone: BackboneFeatureStore,
        channel_factory: t.Optional[t.Callable[[str], CommChannelBase]] = None,
        name: t.Optional[str] = None,
    ) -> None:
        """Initialize the EventPublisher instance.

        :param backbone: The MLI backbone feature store
        :param channel_factory: Factory method to construct new channel instances
        :param name: A unique identifer assigned to the broadcaster for logging. If
         not provided, the system will auto-assign one.
        """
        self._backbone = backbone
        """The backbone feature store used to retrieve consumer descriptors"""
        self._channel_factory = channel_factory
        """A factory method used to instantiate channels from descriptors"""
        self._channel_cache: t.Dict[str, t.Optional[CommChannelBase]] = defaultdict(
            lambda: None
        )
        """A mapping of instantiated channels that can be re-used. Automatically 
        calls the channel factory if a descriptor is not already in the collection"""
        self._event_buffer: t.Deque[EventBase] = deque()
        """A buffer for storing events when a consumer list is not found"""
        self._descriptors: t.Set[str]
        """Stores the most recent list of broadcast consumers. Updated automatically
        on each broadcast"""
        self._name = name or str(uuid.uuid4())
        """A unique identifer assigned to the broadcaster for logging"""

    @property
    def name(self) -> str:
        """The friendly name assigned to the broadcaster.

        :returns: The broadcaster name if one is assigned, otherwise a unique
        id assigned by the system.
        """
        return self._name

    @property
    def num_buffered(self) -> int:
        """Return the number of events currently buffered to send.

        :returns: Number of buffered events
        """
        return len(self._event_buffer)

    def _save_to_buffer(self, event: EventBase) -> None:
        """Places the event in the buffer to be sent once a consumer
        list is available.

        :param event: The event to buffer
        :raises ValueError: If the event cannot be buffered
        """
        try:
            self._event_buffer.append(event)
            logger.debug(f"Buffered event {event=}")
        except Exception as ex:
            raise ValueError(
                f"Unable to buffer event {event} in broadcaster {self.name}"
            ) from ex

    def _log_broadcast_start(self) -> None:
        """Logs broadcast statistics."""
        num_events = len(self._event_buffer)
        num_copies = len(self._descriptors)
        logger.debug(
            f"Broadcast {num_events} events to {num_copies} consumers from {self.name}"
        )

    def _prune_unused_consumers(self) -> None:
        """Performs maintenance on the channel cache by pruning any channel
        that has been removed from the consumers list."""
        active_consumers = set(self._descriptors)
        current_channels = set(self._channel_cache.keys())

        # find any cached channels that are now unused
        inactive_channels = current_channels.difference(active_consumers)
        new_channels = active_consumers.difference(current_channels)

        for descriptor in inactive_channels:
            self._channel_cache.pop(descriptor)

        logger.debug(
            f"Pruning {len(inactive_channels)} stale consumers and"
            f" found {len(new_channels)} new channels for {self.name}"
        )

    def _get_comm_channel(self, descriptor: str) -> CommChannelBase:
        """Helper method to build and cache a comm channel.

        :param descriptor: The descriptor to pass to the channel factory
        :returns: The instantiated channel
        :raises SmartSimError: If the channel fails to attach
        """
        comm_channel = self._channel_cache[descriptor]
        if comm_channel is not None:
            return comm_channel

        if self._channel_factory is None:
            raise SmartSimError("No channel factory provided for consumers")

        try:
            channel = self._channel_factory(descriptor)
            self._channel_cache[descriptor] = channel
            return channel
        except Exception as ex:
            msg = f"Unable to construct channel with descriptor: {descriptor}"
            logger.error(msg, exc_info=True)
            raise SmartSimError(msg) from ex

    def _get_next_event(self) -> t.Optional[EventBase]:
        """Pop the next event to be sent from the queue.

        :returns: The next event to send if any events are enqueued, otherwise `None`.
        """
        try:
            return self._event_buffer.popleft()
        except IndexError:
            logger.debug(f"Broadcast buffer exhausted for {self.name}")

        return None

    def _broadcast(self, timeout: float = 0.001) -> BroadcastResult:
        """Broadcasts all buffered events to registered event consumers.

        :param timeout: Maximum time to wait (in seconds) for messages to send
        :returns: BroadcastResult containing the number of messages that were
        successfully and unsuccessfully sent for all consumers
        :raises SmartSimError: If the channel fails to attach or broadcasting fails
        """
        # allow descriptors to be empty since events are buffered
        self._descriptors = set(x for x in self._backbone.notification_channels if x)
        if not self._descriptors:
            msg = f"No event consumers are registered for {self.name}"
            logger.warning(msg)
            return BroadcastResult(0, 0)

        self._prune_unused_consumers()
        self._log_broadcast_start()

        num_listeners = len(self._descriptors)
        num_sent = 0
        num_failures = 0

        # send each event to every consumer
        while event := self._get_next_event():
            logger.debug(f"Broadcasting {event=} to {num_listeners} listeners")
            event_bytes = bytes(event)

            for i, descriptor in enumerate(self._descriptors):
                comm_channel = self._get_comm_channel(descriptor)

                try:
                    comm_channel.send(event_bytes, timeout)
                    num_sent += 1
                except Exception:
                    msg = (
                        f"Broadcast {i+1}/{num_listeners} for event {event.uid} to "
                        f"channel {descriptor} from {self.name} failed."
                    )
                    logger.exception(msg)
                    num_failures += 1

        return BroadcastResult(num_sent, num_failures)

    def send(self, event: EventBase, timeout: float = 0.001) -> int:
        """Implementation of `send` method of the `EventPublisher` protocol. Publishes
        the supplied event to all registered broadcast consumers.

        :param event: An event to publish
        :param timeout: Maximum time to wait (in seconds) for messages to send
        :returns: The total number of events successfully published to consumers
        :raises ValueError: If event serialization fails
        :raises AttributeError: If event cannot be serialized
        :raises KeyError: If channel fails to attach using registered descriptors
        :raises SmartSimError: If any unexpected error occurs during send
        """
        try:
            self._save_to_buffer(event)
            result = self._broadcast(timeout)
            return result.num_sent
        except (KeyError, ValueError, AttributeError, SmartSimError):
            raise
        except Exception as ex:
            raise SmartSimError("An unexpected failure occurred while sending") from ex
