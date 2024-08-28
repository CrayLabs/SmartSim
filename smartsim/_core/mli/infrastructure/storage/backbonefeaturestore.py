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

import enum
import pickle
import typing as t
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass

# pylint: disable=import-error
# isort: off
import dragon.data.ddict.ddict as dragon_ddict

# isort: on

from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.featurestore import ReservedKeys
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


# todo: did i create an arms race where a developer just grabs the backbone
# and passes it wherever they need a FeatureStore?
class BackboneFeatureStore(DragonFeatureStore):
    """A DragonFeatureStore wrapper with utility methods for accessing shared
    information stored in the MLI backbone feature store"""

    def __init__(self, storage: "dragon_ddict.DDict") -> None:
        """Initialize the DragonFeatureStore instance

        :param storage: A distributed dictionary to be used as the underlying
        storage mechanism of the feature store"""
        super().__init__(storage)
        self._reserved_write_enabled = True

    @property
    def notification_channels(self) -> t.Sequence[str]:
        """Retrieve descriptors for all registered MLI notification channels

        :returns: the list of descriptors"""
        if ReservedKeys.MLI_NOTIFY_CONSUMERS in self:
            stored_consumers = self[ReservedKeys.MLI_NOTIFY_CONSUMERS]
            return str(stored_consumers).split(",")
        return []

    @notification_channels.setter
    def notification_channels(self, values: t.Sequence[str]) -> None:
        """Set the notification channels to be sent events

        :param values: the list of channel descriptors to save"""
        self[ReservedKeys.MLI_NOTIFY_CONSUMERS] = ",".join(
            [str(value) for value in values]
        )


class EventCategory(str, enum.Enum):
    """Predefined event types raised by SmartSim backend"""

    CONSUMER_CREATED: str = "consumer-created"
    FEATURE_STORE_WRITTEN: str = "feature-store-written"
    UNKNOWN: str = "unknown"


@dataclass
class EventBase:
    """Core API for an event"""

    category: EventCategory
    """The event category for this event; may be used for addressing,
    prioritization, or filtering of events by a event publisher/consumer"""

    uid: str
    """A unique identifier for this event"""

    def __bytes__(self) -> bytes:
        """Default conversion to bytes for an event required to publish
        messages using byte-oriented communication channels

        :returns: this entity encoded as bytes"""
        return pickle.dumps(self)

    def __str__(self) -> str:
        """Convert the event to a string

        :returns: a string representation of this instance"""
        return f"{self.uid}|{self.category}"


class OnCreateConsumer(EventBase):
    """Publish this event when a new event consumer registration is required"""

    descriptor: str
    """Descriptor of the comm channel exposed by the consumer"""

    def __init__(self, descriptor: str) -> None:
        """Initialize the event

        :param descriptor: descriptor of the comm channel exposed by the consumer
        """
        super().__init__(EventCategory.CONSUMER_CREATED, str(uuid.uuid4()))
        self.descriptor = descriptor

    def __str__(self) -> str:
        """Convert the event to a string

        :returns: a string representation of this instance"""
        return f"{str(super())}|{self.descriptor}"


class OnWriteFeatureStore(EventBase):
    """Publish this event when a feature store key is written"""

    descriptor: str
    """The descriptor of the feature store where the write occurred"""

    key: str
    """The key identifying where the write occurred"""

    def __init__(self, descriptor: str, key: str) -> None:
        """Initialize the event

        :param descriptor: The descriptor of the feature store where the write occurred
        :param key: The key identifying where the write occurred
        """
        super().__init__(EventCategory.FEATURE_STORE_WRITTEN, str(uuid.uuid4()))
        self.descriptor = descriptor
        self.key = key

    def __str__(self) -> str:
        """Convert the event to a string

        :returns: a string representation of this instance"""
        return f"{str(super())}|{self.descriptor}|{self.key}"


class EventPublisher(t.Protocol):
    """Core API of a class that publishes events"""

    def send(self, event: EventBase) -> int:
        """The send operation"""


class EventBroadcaster:
    """Performs fan-out publishing of system events"""

    def __init__(
        self,
        backbone: BackboneFeatureStore,
        channel_factory: t.Optional[t.Callable[[str], CommChannelBase]] = None,
    ) -> None:
        """Initialize the EventPublisher instance

        :param backbone: the MLI backbone feature store
        :param channel_factory: factory method to construct new channel instances
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
        self._event_buffer: t.Deque[bytes] = deque()
        """A buffer for storing events when a consumer list is not found."""
        self._descriptors: t.Set[str]
        """Stores the most recent list of broadcast consumers. Updated automatically
        on each broadcast"""

    @property
    def num_buffered(self) -> int:
        """Return the number of events currently buffered to send"""
        return len(self._event_buffer)

    def _save_to_buffer(self, event: EventBase) -> None:
        """Places a serialized event in the buffer to be sent once a consumer
        list is available.

        :param event: The event to serialize and buffer"""

        try:
            event_bytes = bytes(event)
            self._event_buffer.append(event_bytes)
        except Exception as ex:
            raise ValueError("Unable to serialize event for sending") from ex

    def _log_broadcast_start(self) -> None:
        """Logs broadcast statistics"""
        num_pending = len(self._event_buffer)
        num_consumers = len(self._descriptors)
        logger.debug(f"Broadcasting {num_pending} events to {num_consumers} consumers")

    def _prune_unused_consumers(self) -> None:
        """Performs maintenance on the channel cache by pruning any channel
        that has been removed from the consumers list"""
        active_consumers = set(self._descriptors)
        current_channels = set(self._channel_cache.keys())

        # find any cached channels that are now unused
        inactive_channels = current_channels.difference(active_consumers)
        new_channels = active_consumers.difference(current_channels)

        for descriptor in inactive_channels:
            self._channel_cache.pop(descriptor)

        logger.debug(
            f"Pruning {len(inactive_channels)} stale consumer channels"
            f" and found {len(new_channels)} new channels"
        )

    def _get_comm_channel(self, descriptor: str) -> CommChannelBase:
        """Helper method to build and cache a comm channel

        :param descriptor: the descriptor to pass to the channel factory
        :returns: the instantiated channel
        :raises SmartSimError: if the channel fails to build"""
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

    def _broadcast(self) -> int:
        """Broadcasts all buffered events to registered event consumers.

        :return: the number of events broadcasted to consumers
        :raises ValueError: if event serialization fails
        :raises KeyError: if channel fails to attach using registered descriptors
        :raises SmartSimError: if broadcasting fails"""

        # allow descriptors to be empty since events are buffered
        self._descriptors = set(x for x in self._backbone.notification_channels if x)
        if not self._descriptors:
            logger.warning("No event consumers are registered")
            return 0

        self._prune_unused_consumers()
        self._log_broadcast_start()

        num_sent: int = 0
        next_event: t.Optional[bytes] = self._event_buffer.popleft()

        # send each event to every consumer
        while next_event is not None:
            for descriptor in map(str, self._descriptors):
                comm_channel = self._get_comm_channel(descriptor)

                try:
                    comm_channel.send(next_event)
                    num_sent += 1
                except Exception as ex:
                    raise SmartSimError(
                        f"Failed broadcast to channel: {descriptor}"
                    ) from ex

            try:
                next_event = self._event_buffer.popleft()
            except IndexError:
                next_event = None
                logger.debug("Event buffer exhausted")

        return num_sent

    def send(self, event: EventBase) -> int:
        """Implementation of `send` method of the `EventPublisher` protocol. Publishes
        the supplied event to all registered broadcast consumers

        :param event: an event to publish
        :returns: the number of events successfully published
        :raises ValueError: if event serialization fails
        :raises KeyError: if channel fails to attach using registered descriptors
        :raises SmartSimError: if any unexpected error occurs during send"""
        try:
            self._save_to_buffer(event)

            return self._broadcast()
        except (KeyError, ValueError, SmartSimError):
            raise
        except Exception as ex:
            raise SmartSimError("An unexpected failure occurred while sending") from ex


class EventConsumer:
    """Reads system events published to a communications channel"""

    def __init__(
        self,
        comm_channel: CommChannelBase,
        backbone: BackboneFeatureStore,
        filters: t.Optional[t.List[EventCategory]] = None,
        timeout: int = 0,
    ) -> None:
        """Initialize the EventConsumer instance

        :param comm_channel: communications channel to listen to for events
        :param backbone: the MLI backbone feature store
        :param filters: a list of event types to deliver. when empty, all
        events will be delivered
        :param timeout: maximum time to wait for messages to arrive; may be overridden
        on individual calls to `receive`"""
        self._comm_channel = comm_channel
        self._backbone = backbone
        self._global_filters = filters or []
        self._global_timeout = timeout

    def receive(
        self, filters: t.Optional[t.List[EventCategory]] = None, timeout: int = 0
    ) -> t.List[EventBase]:
        """Receives available published event(s)

        :param filters: additional filters to add to the global filters configured
        on the EventConsumer instance
        :param timeout: maximum time to wait for messages to arrive
        :returns: a list of events that pass any configured filters"""
        if filters is None:
            filters = []

        filter_set = {*self._global_filters, *filters}
        messages: t.List[t.Any] = []

        # use the local timeout to override a global setting
        timeout = timeout or self._global_timeout
        msg_bytes_list = self._comm_channel.recv(timeout)

        # remove any empty messages that will fail to decode
        msg_bytes_list = [msg for msg in msg_bytes_list if msg]

        msg: t.Optional[EventBase] = None
        if msg_bytes_list:
            for message in msg_bytes_list:
                msg = pickle.loads(message)

                if not msg:
                    continue

                # ignore anything that doesn't match a filter (if one is
                # supplied), otherwise return everything
                if not filter_set or msg.category in filter_set:
                    messages.append(msg)

        return messages
