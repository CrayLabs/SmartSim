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
import enum
import itertools
import pickle
import time
import typing as t
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass

# pylint: disable=import-error
# isort: off
import dragon.data.ddict.ddict as dragon_ddict

# isort: on

from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel
from smartsim._core.mli.infrastructure.storage.dragon_feature_store import (
    DragonFeatureStore,
)
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


def byte_descriptor_to_string(descriptor: bytes) -> str:
    return base64.b64encode(descriptor).decode("utf-8")


def string_descriptor_to_byte(descriptor: str) -> bytes:
    return base64.b64decode(descriptor.encode("utf-8"))


# todo: did i create an arms race where a developer just grabs the backbone
# and passes it wherever they need a FeatureStore?
class BackboneFeatureStore(DragonFeatureStore):
    """A DragonFeatureStore wrapper with utility methods for accessing shared
    information stored in the MLI backbone feature store."""

    MLI_NOTIFY_CONSUMERS = "_SMARTSIM_MLI_NOTIFY_CONSUMERS"
    MLI_BACKEND_CONSUMER = "_SMARTIM_MLI_BACKEND_CONSUMER"
    MLI_WORKER_QUEUE = "to_worker_fli"

    def __init__(
        self,
        storage: "dragon_ddict.DDict",
        allow_reserved_writes: bool = False,
    ) -> None:
        """Initialize the DragonFeatureStore instance.

        :param storage: A distributed dictionary to be used as the underlying
        storage mechanism of the feature store
        :param allow_reserved_writes: Whether reserved writes are allowed
        """
        super().__init__(storage)
        self._enable_reserved_writes = allow_reserved_writes

    @property
    def notification_channels(self) -> t.Sequence[str]:
        """Retrieve descriptors for all registered MLI notification channels.

        :returns: The list of descriptors
        """
        if "_SMARTSIM_MLI_NOTIFY_CONSUMERS" in self:
            stored_consumers = self[self.MLI_NOTIFY_CONSUMERS]
            return str(stored_consumers).split(",")
        return []

    @notification_channels.setter
    def notification_channels(self, values: t.Sequence[str]) -> None:
        """Set the notification channels to be sent events.

        :param values: The list of channel descriptors to save
        """
        self[self.MLI_NOTIFY_CONSUMERS] = ",".join([str(value) for value in values])

    @property
    def backend_channel(self) -> t.Optional[str]:
        """Retrieve the channel descriptor exposed by the MLI backend for events

        :returns: a stringified channel descriptor"""
        if self.MLI_NOTIFY_CONSUMERS in self:
            return str(self[self.MLI_NOTIFY_CONSUMERS])
        return None

    @backend_channel.setter
    def backend_channel(self, value: str) -> None:
        """Set the channel exposed by the MLI backend for events

        :param value: a stringified channel descriptor"""
        self[self.MLI_NOTIFY_CONSUMERS] = value

    @property
    def worker_queue(self) -> t.Optional[str]:
        """Retrieve the channel descriptor exposed by the MLI
        backend to send work to an MLI worker manager instance

        :returns: a stringified channel descriptor"""
        if self.MLI_WORKER_QUEUE in self:
            return str(self[self.MLI_WORKER_QUEUE])
        return None

    @worker_queue.setter
    def worker_queue(self, value: str) -> None:
        """Set the channel descriptor exposed by the MLI
        backend to send work to an MLI worker manager instance

        :param value: a stringified channel descriptor"""
        self[self.MLI_WORKER_QUEUE] = value

    @classmethod
    def from_writable_descriptor(
        cls,
        descriptor: str,
    ) -> "BackboneFeatureStore":
        """A factory method that creates an instance from a descriptor string

        :param descriptor: The descriptor that uniquely identifies the resource
        :returns: An attached DragonFeatureStore
        :raises SmartSimError: if attachment to DragonFeatureStore fails"""
        try:
            return BackboneFeatureStore(dragon_ddict.DDict.attach(descriptor), True)
        except Exception as ex:
            logger.error(f"Error creating dragon feature store: {descriptor}")
            raise SmartSimError(
                f"Error creating dragon feature store: {descriptor}"
            ) from ex

    @staticmethod
    def _check_wait_timeout(
        start_time: float, timeout: float, indicators: t.Dict[str, bool]
    ) -> None:
        """Perform timeout verification

        :param start_time: the start time to use for elapsed calculation
        :param timeout: the timeout (in seconds)
        :param indicators: latest retrieval status for requested keys"""
        elapsed = time.time() - start_time
        if timeout and elapsed > timeout:
            raise SmartSimError(
                f"Timeout retrieving all keys from backbone: {indicators}"
            )

    def wait_for(
        self, keys: t.List[str], timeout: float = 0
    ) -> t.Dict[str, t.Union[str, bytes, None]]:
        """Perform a blocking wait until all specified keys have been found
        in the backbone

        :param keys: The required collection of keys to retrieve
        :param timeout: The maximum wait time in seconds. Overrides class level setting
        """

        to_check = list(keys)
        was_found = [False for _ in to_check]  # add test ensuring dupes are handled..
        values: t.List[t.Union[str, bytes, None]] = [None for _ in to_check]

        backoff = [0.1, 0.5, 1, 2, 4, 8]
        backoff_iter = itertools.cycle(backoff)
        start_time = time.time()

        while not all(was_found):
            delay = next(backoff_iter)

            for index, key in enumerate(to_check):
                if was_found[index]:
                    continue

                try:
                    values[index] = self[key]
                    was_found[index] = True
                except KeyError:
                    logger.debug(f"Re-attempting `{key}` retrieval in {delay}s")

            if all(was_found):
                continue

            self._check_wait_timeout(
                start_time, timeout, dict(zip(to_check, was_found))
            )

            time.sleep(delay)

        return dict(zip(keys, values))


class EventCategory(str, enum.Enum):
    """Predefined event types raised by SmartSim backend."""

    CONSUMER_CREATED: str = "consumer-created"
    FEATURE_STORE_WRITTEN: str = "feature-store-written"


@dataclass
class EventBase:
    """Core API for an event."""

    # todo: shift eventing code to: infrastructure / event / event.py
    category: EventCategory
    """The event category for this event; may be used for addressing,
    prioritization, or filtering of events by a event publisher/consumer"""

    uid: str
    """A unique identifier for this event"""

    def __bytes__(self) -> bytes:
        """Default conversion to bytes for an event required to publish
        messages using byte-oriented communication channels.

        :returns: This entity encoded as bytes"""
        return pickle.dumps(self)

    def __str__(self) -> str:
        """Convert the event to a string.

        :returns: A string representation of this instance"""
        return f"{self.uid}|{self.category}"


class OnCreateConsumer(EventBase):
    """Publish this event when a new event consumer registration is required."""

    descriptor: str
    """Descriptor of the comm channel exposed by the consumer"""
    filters: t.List[EventCategory]
    """The collection of filters indicating messages of interest to this consumer"""

    def __init__(self, descriptor: str, filters: t.Sequence[EventCategory]) -> None:
        """Initialize the OnCreateConsumer event.

        :param descriptor: Descriptor of the comm channel exposed by the consumer
        :param descriptor: Collection of filters indicating messages of interest
        """
        super().__init__(EventCategory.CONSUMER_CREATED, str(uuid.uuid4()))
        self.descriptor = descriptor
        self.filters = list(filters)

    def __str__(self) -> str:
        """Convert the event to a string.

        :returns: A string representation of this instance
        """
        _filters = ",".join(self.filters)
        return f"{str(super())}|{self.descriptor}|{_filters}"


class OnWriteFeatureStore(EventBase):
    """Publish this event when a feature store key is written."""

    descriptor: str
    """The descriptor of the feature store where the write occurred"""

    key: str
    """The key identifying where the write occurred"""

    def __init__(self, descriptor: str, key: str) -> None:
        """Initialize the OnWriteFeatureStore event.

        :param descriptor: The descriptor of the feature store where the write occurred
        :param key: The key identifying where the write occurred
        """
        super().__init__(EventCategory.FEATURE_STORE_WRITTEN, str(uuid.uuid4()))
        self.descriptor = descriptor
        self.key = key

    def __str__(self) -> str:
        """Convert the event to a string.

        :returns: A string representation of this instance
        """
        return f"{str(super())}|{self.descriptor}|{self.key}"


class EventProducer(t.Protocol):
    """Core API of a class that publishes events."""

    def send(self, event: EventBase, timeout: float = 0.001) -> int:
        """The send operation.

        :param event: The event to send
        :param timeout: Maximum time to wait (in seconds) for messages to send
        """


class EventSender:
    """An event publisher that performs publishing of system events to a
    single endpoint"""

    def __init__(
        self,
        backbone: BackboneFeatureStore,
        channel: t.Optional[CommChannelBase],
    ) -> None:
        """Initialize the instance"""
        self._backbone = backbone
        self._channel: t.Optional[CommChannelBase] = channel

    def send(self, event: EventBase) -> int:
        """The send operation"""
        if self._channel is None:
            # self._channel = self._channel_factory(event)
            raise Exception("No channel to send on")
        num_sent = 0

        try:
            event_bytes = bytes(event)
            self._channel.send(event_bytes)
            num_sent += 1
        except Exception as ex:
            raise SmartSimError(f"Failed broadcast to channel: {self._channel}") from ex

        return num_sent


class EventBroadcaster:
    """Performs fan-out publishing of system events."""

    def __init__(
        self,
        backbone: BackboneFeatureStore,
        channel_factory: t.Optional[t.Callable[[str], CommChannelBase]] = None,
    ) -> None:
        """Initialize the EventPublisher instance.

        :param backbone: The MLI backbone feature store
        :param channel_factory: Factory method to construct new channel instances
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
        """A buffer for storing events when a consumer list is not found"""
        self._descriptors: t.Set[str]
        """Stores the most recent list of broadcast consumers. Updated automatically
        on each broadcast"""
        self._uid = str(uuid.uuid4())
        """A unique identifer assigned to the broadcaster for logging"""

    @property
    def num_buffered(self) -> int:
        """Return the number of events currently buffered to send.

        :returns: Number of buffered events
        """
        return len(self._event_buffer)

    def _save_to_buffer(self, event: EventBase) -> None:
        """Places a serialized event in the buffer to be sent once a consumer
        list is available.

        :param event: The event to serialize and buffer
        :raises ValueError: If the event cannot be serialized
        """
        try:
            event_bytes = bytes(event)
            self._event_buffer.append(event_bytes)
        except Exception as ex:
            raise ValueError(f"Unable to serialize event from {self._uid}") from ex

    def _log_broadcast_start(self) -> None:
        """Logs broadcast statistics."""
        num_events = len(self._event_buffer)
        num_copies = len(self._descriptors)
        logger.debug(
            f"Broadcast {num_events} events to {num_copies} consumers from {self._uid}"
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
            f" found {len(new_channels)} new channels for {self._uid}"
        )

    def _get_comm_channel(self, descriptor: str) -> CommChannelBase:
        """Helper method to build and cache a comm channel.

        :param descriptor: The descriptor to pass to the channel factory
        :returns: The instantiated channel
        :raises SmartSimError: If the channel fails to build
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

    def _broadcast(self, timeout: float = 0.001) -> int:
        """Broadcasts all buffered events to registered event consumers.

        :param timeout: Maximum time to wait (in seconds) for messages to send
        :returns: The number of events broadcasted to consumers
        :raises SmartSimError: If broadcasting fails
        """
        # allow descriptors to be empty since events are buffered
        self._descriptors = set(x for x in self._backbone.notification_channels if x)
        if not self._descriptors:
            logger.warning(f"No event consumers are registered for {self._uid}")
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
                    # todo: given a failure, the message is not sent to any other
                    # recipients. consider retrying, adding a dead letter queue, or
                    # logging the message details more intentionally
                    comm_channel.send(next_event, timeout)
                    num_sent += 1
                except Exception as ex:
                    raise SmartSimError(
                        f"Failed broadcast to channel {descriptor} from {self._uid}"
                    ) from ex

            try:
                next_event = self._event_buffer.popleft()
            except IndexError:
                next_event = None
                logger.debug(f"Broadcast buffer exhausted for {self._uid}")

        return num_sent

    def send(self, event: EventBase, timeout: float = 0.001) -> int:
        """Implementation of `send` method of the `EventPublisher` protocol. Publishes
        the supplied event to all registered broadcast consumers.

        :param event: An event to publish
        :param timeout: Maximum time to wait (in seconds) for messages to send
        :returns: The number of events successfully published
        :raises ValueError: If event serialization fails
        :raises KeyError: If channel fails to attach using registered descriptors
        :raises SmartSimError: If any unexpected error occurs during send
        """
        try:
            self._save_to_buffer(event)
            return self._broadcast(timeout)
        except (KeyError, ValueError, SmartSimError):
            raise
        except Exception as ex:
            raise SmartSimError("An unexpected failure occurred while sending") from ex


class EventConsumer:
    """Reads system events published to a communications channel."""

    def __init__(
        self,
        comm_channel: CommChannelBase,
        backbone: BackboneFeatureStore,
        filters: t.Optional[t.List[EventCategory]] = None,
        batch_timeout: t.Optional[float] = None,
        name: t.Optional[str] = None,
    ) -> None:
        """Initialize the EventConsumer instance.

        :param comm_channel: Communications channel to listen to for events
        :param backbone: The MLI backbone feature store
        :param filters: A list of event types to deliver. when empty, all
        events will be delivered
        :param timeout: Maximum time to wait for messages to arrive; may be overridden
        on individual calls to `receive`
        :raises ValueError: If batch_timeout <= 0
        """
        if batch_timeout is not None and batch_timeout <= 0:
            raise ValueError("batch_timeout must be a non-zero, positive value")

        self._comm_channel = comm_channel
        self._backbone = backbone
        self._global_filters = filters or []
        self._global_timeout = batch_timeout or 1.0
        self._name = name

    def receive(
        self, filters: t.Optional[t.List[EventCategory]] = None, timeout: float = 0
    ) -> t.List[EventBase]:
        """Receives available published event(s).

        :param filters: Additional filters to add to the global filters configured
        on the EventConsumer instance
        :param timeout: Maximum time to wait for messages to arrive
        :returns: A list of events that pass any configured filters
        """
        if filters is None:
            filters = []

        filter_set = {*self._global_filters, *filters}
        messages: t.List[t.Any] = []

        # use the local timeout to override a global setting
        start_at = time.time_ns()

        while msg_bytes_list := self._comm_channel.recv(timeout=timeout):
            # remove any empty messages that will fail to decode
            msg_bytes_list = [msg for msg in msg_bytes_list if msg]

            msg: t.Optional[EventBase] = None
            if msg_bytes_list:
                for message in msg_bytes_list:
                    msg = pickle.loads(message)

                    if not msg:
                        logger.warning("Unable to unpickle message")
                        continue

                    # ignore anything that doesn't match a filter (if one is
                    # supplied), otherwise return everything
                    if not filter_set or msg.category in filter_set:
                        messages.append(msg)

            # avoid getting stuck indefinitely waiting for the channel
            elapsed = (time.time_ns() - start_at) / 1000000000
            remaining = elapsed - self._global_timeout
            if remaining > 0:
                logger.debug(f"consumer batch timeout exceeded by: {abs(remaining)}")
                break

        return messages

    def register(self) -> t.Generator[bool, None, None]:
        """Send an event to register this consumer as a listener"""
        awaiting_confirmation = True
        descriptor = self._comm_channel.descriptor
        backoffs = itertools.cycle((0.1, 0.5, 1.0, 2.0, 4.0, 8.0))
        event = OnCreateConsumer(descriptor, self._global_filters)

        while awaiting_confirmation:
            registered_channels = self._backbone.notification_channels
            # todo: this should probably be descriptor_string? maybe i need to
            # get rid of descriptor as bytes or just make desc_string required in ABC
            if descriptor in registered_channels:
                awaiting_confirmation = False

            yield not awaiting_confirmation
            time.sleep(next(backoffs))

            if backend_descriptor := self._backbone.backend_channel:
                backend_channel = DragonCommChannel.from_descriptor(backend_descriptor)
                backend = EventSender(self._backbone, backend_channel)
                backend.send(event)

    # def register_callback(self, callback: t.Callable[[EventBase], None]) -> None: ...

    def listen(self, fn: t.Callable[[EventBase], None]) -> None:
        """Function to handle incoming events"""
        while True:
            incoming_messages = self.receive()
            for message in incoming_messages:
                fn(message)
