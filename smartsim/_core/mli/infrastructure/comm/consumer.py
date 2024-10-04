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

import pickle
import time
import typing as t
import uuid

from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel
from smartsim._core.mli.infrastructure.comm.event import (
    EventBase,
    OnCreateConsumer,
    OnRemoveConsumer,
    OnShutdownRequested,
)
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim.log import get_logger

logger = get_logger(__name__)


class EventConsumer:
    """Reads system events published to a communications channel."""

    _BACKBONE_WAIT_TIMEOUT = 10.0
    """Maximum time (in seconds) to wait for the backbone to register the consumer"""

    def __init__(
        self,
        comm_channel: CommChannelBase,
        # channel_factory: ...,
        backbone: BackboneFeatureStore,
        filters: t.Optional[t.List[str]] = None,
        name: t.Optional[str] = None,
        event_handler: t.Optional[t.Callable[[EventBase], None]] = None,
    ) -> None:
        """Initialize the EventConsumer instance.

        :param comm_channel: Communications channel to listen to for events
        :param backbone: The MLI backbone feature store
        :param filters: A list of event types to deliver. when empty, all
        events will be delivered
        :param name: A user-friendly name for logging. If not provided, an
        auto-generated GUID will be used
        :raises ValueError: If batch_timeout <= 0
        """
        self._comm_channel = comm_channel
        """The comm channel used by the consumer to receive messages. The channel
        descriptor will be published for senders to discover."""
        self._backbone = backbone
        """The backbone instance used to bootstrap the instance. The EventConsumer
        uses the backbone to discover where it can publish its descriptor."""
        self._global_filters = filters or []
        """A set of global filters to apply to incoming events. Global filters are
        combined with per-call filters. Filters act as an allow-list."""
        self._name = name or str(uuid.uuid4())
        """User-friendly name assigned to a consumer for logging. Automatically
        assigned if not provided."""
        self._event_handler = event_handler
        """The function that should be executed when an event
        passed by the filters is received."""
        self.listening = True
        """Flag indicating that the consumer is currently listening for new
        events. Setting this flag to `False` will cause any active calls to
        `listen` to terminate."""

    @property
    def descriptor(self) -> str:
        """The descriptor of the underlying comm channel.

        :returns: The comm channel descriptor"""
        return self._comm_channel.descriptor

    @property
    def name(self) -> str:
        """The friendly name assigned to the consumer.

        :returns: The consumer name if one is assigned, otherwise a unique
        id assigned by the system.
        """
        return self._name

    def recv(
        self,
        filters: t.Optional[t.List[str]] = None,
        timeout: float = 0.001,
        batch_timeout: float = 1.0,
    ) -> t.List[EventBase]:
        """Receives available published event(s).

        :param filters: Additional filters to add to the global filters configured
        on the EventConsumer instance
        :param timeout: Maximum time to wait for a single message to arrive
        :param batch_timeout: Maximum time to wait for messages to arrive; allows
        multiple batches to be retrieved in one call to `send`
        :returns: A list of events that pass any configured filters
        :raises ValueError: If a positive, non-zero value is not provided for the
        timeout or batch_timeout.
        """
        if filters is None:
            filters = []

        if timeout is not None and timeout <= 0:
            raise ValueError("request timeout must be a non-zero, positive value")

        if batch_timeout is not None and batch_timeout <= 0:
            raise ValueError("batch_timeout must be a non-zero, positive value")

        filter_set = {*self._global_filters, *filters}
        all_message_bytes: t.List[bytes] = []

        # firehose as many messages as possible within the batch_timeout
        start_at = time.time()
        remaining = batch_timeout

        batch_message_bytes = self._comm_channel.recv(timeout=timeout)
        while batch_message_bytes:
            # remove any empty messages that will fail to decode
            all_message_bytes.extend(batch_message_bytes)
            batch_message_bytes = []

            # avoid getting stuck indefinitely waiting for the channel
            elapsed = time.time() - start_at
            remaining = batch_timeout - elapsed

            if remaining > 0:
                batch_message_bytes = self._comm_channel.recv(timeout=timeout)

        events_received: t.List[EventBase] = []

        # Timeout elapsed or no messages received - return the empty list
        if not all_message_bytes:
            return events_received

        for message in all_message_bytes:
            if not message or message is None:
                continue

            event = pickle.loads(message)
            if not event:
                logger.warning(f"Consumer {self.name} is unable to unpickle message")
                continue

            # skip events that don't pass a filter
            if filter_set and event.category not in filter_set:
                continue

            events_received.append(event)

        return events_received

    def _send_to_registrar(self, event: EventBase) -> None:
        """Send an event direct to the registrar listener."""
        registrar_key = BackboneFeatureStore.MLI_REGISTRAR_CONSUMER
        config = self._backbone.wait_for([registrar_key], self._BACKBONE_WAIT_TIMEOUT)
        registrar_descriptor = str(config.get(registrar_key, None))

        if not registrar_descriptor:
            logger.warning(
                f"Unable to send {event.category} from {self.name}. "
                "No registrar channel found."
            )
            return

        logger.debug(f"Sending {event.category} from {self.name}")

        registrar_channel = DragonCommChannel.from_descriptor(registrar_descriptor)
        registrar_channel.send(bytes(event), timeout=1.0)

        logger.debug(f"{event.category} from {self.name} sent")

    def register(self) -> None:
        """Send an event to register this consumer as a listener."""
        descriptor = self._comm_channel.descriptor
        event = OnCreateConsumer(self.name, descriptor, self._global_filters)

        self._send_to_registrar(event)

    def unregister(self) -> None:
        """Send an event to un-register this consumer as a listener."""
        descriptor = self._comm_channel.descriptor
        event = OnRemoveConsumer(self.name, descriptor)

        self._send_to_registrar(event)

    def _on_handler_missing(self, event: EventBase) -> None:
        """A "dead letter" event handler that is called to perform
        processing on events before they're discarded.

        :param event: The event to handle
        """
        logger.warning(
            "No event handler is registered in consumer "
            f"{self.name}. Discarding {event=}"
        )

    def listen_once(self, timeout: float = 0.001, batch_timeout: float = 1.0) -> None:
        """Receives messages for the consumer a single time. Delivers
        all messages that pass the consumer filters. Shutdown requests
        are handled by a default event handler.


        NOTE: Executes a single batch-retrieval to receive the maximum
        number of messages available under batch timeout. To continually
        listen, use `listen` in a non-blocking thread/process

        :param timeout: Maximum time to wait (in seconds) for a message to arrive
        :param timeout: Maximum time to wait (in seconds) for a batch to arrive
        """
        logger.info(
            f"Consumer {self.name} listening with {timeout} second timeout"
            f" on channel {self._comm_channel.descriptor}"
        )

        if not self._event_handler:
            logger.info("Unable to handle messages. No event handler is registered.")

        incoming_messages = self.recv(timeout=timeout, batch_timeout=batch_timeout)

        if not incoming_messages:
            logger.info(f"Consumer {self.name} received empty message list")

        for message in incoming_messages:
            logger.info(f"Consumer {self.name} is handling event {message=}")
            self._handle_shutdown(message)

            if self._event_handler:
                self._event_handler(message)
            else:
                self._on_handler_missing(message)

    def _handle_shutdown(self, event: EventBase) -> bool:
        """Handles shutdown requests sent to the consumer by setting the
        `self.listener` property to `False`.

        :param event: The event to handle
        :returns: A bool indicating if the event was a shutdown request
        """
        if isinstance(event, OnShutdownRequested):
            logger.debug(f"Shutdown requested from: {event.source}")
            self.listening = False
            return True
        return False

    def listen(self, timeout: float = 0.001, batch_timeout: float = 1.0) -> None:
        """Receives messages for the consumer until a shutdown request is received.

        :param timeout: Maximum time to wait (in seconds) for a message to arrive
        :param batch_timeout: Maximum time to wait (in seconds) for a batch to arrive
        """

        logger.debug(f"Consumer {self.name} is now listening for events.")

        while self.listening:
            self.listen_once(timeout, batch_timeout)

        logger.debug(f"Consumer {self.name} is no longer listening.")
