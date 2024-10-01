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
# pylint: disable=import-error
# pylint: disable=unused-import
import dragon

# from dragon.globalservices.api_setup import connect_to_infrastructure


# pylint: enable=unused-import
# pylint: enable=import-error
# isort: on

import argparse
import multiprocessing as mp
import os
import sys
import typing as t

from smartsim._core.entrypoints.service import Service
from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragon_util import create_local
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
    EventBase,
    EventCategory,
    EventConsumer,
    OnCreateConsumer,
    OnRemoveConsumer,
)
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class ConsumerRegistrationListener(Service):
    """A long-running service that listens for events of a specific type
    and executes the appropriate event handler."""

    def __init__(
        self,
        backbone: BackboneFeatureStore,
        timeout: float,
        batch_timeout: float,
        as_service: bool = False,
        cooldown: int = 0,
        health_check_frequency: float = 60.0,
    ) -> None:
        """Initialize the EventListener.

        :param backbone: The backbone feature store
        :param timeout: Maximum time (in seconds) to allow a single recv request to wait
        :param batch_timeout: Maximum time (in seconds) to allow a batch of receives to
         continue to build
        :param filters: Filters specifying the message types to handle
        :param as_service: Specifies run-once or run-until-complete behavior of service
        :param cooldown: Number of seconds to wait before shutting down after
        shutdown criteria are met
        """
        super().__init__(
            as_service, cooldown, health_check_frequency=health_check_frequency
        )

        self._timeout = timeout
        """ Maximum time (in seconds) to allow a single recv request to wait"""

        self._batch_timeout = batch_timeout
        """Maximum time (in seconds) to allow a batch of receives to
         continue to build"""

        self._consumer: t.Optional[EventConsumer] = None
        """The event consumer that handles receiving events"""

        self._backbone = backbone
        """A standalone, system-created feature store used to share internal
        information among MLI components"""

    def _on_start(self) -> None:
        """Called on initial entry into Service `execute` event loop before
        `_on_iteration` is invoked."""
        super()._on_start()
        self._create_eventing()

    def _on_shutdown(self) -> None:
        """Release dragon resources. Called immediately after exiting
        the main event loop during automatic shutdown."""
        super()._on_shutdown()

        # unregister this listener in the backbone
        self._backbone.pop(BackboneFeatureStore.MLI_BACKEND_CONSUMER)

        # TODO: need the channel to be cleaned up
        # self._consumer._comm_channel._channel.destroy()

    def _on_iteration(self) -> None:
        """Executes calls to the machine learning worker implementation to complete
        the inference pipeline."""

        if self._consumer is None:
            logger.info("Unable to listen. No consumer available.")
            return

        self._consumer.listen_once(self._timeout, self._batch_timeout)

    def _can_shutdown(self) -> bool:
        """Determines if the event consumer is ready to stop listening.

        :returns: True when criteria to shutdown the service are met, False otherwise
        """

        if self._backbone is None:
            logger.info("Listener must shutdown: no backbone attached")
            return True

        if self._consumer is None:
            logger.info("Listener must shutdown: no consumer channel created")
            return True

        if not self._consumer.listening:
            logger.info("Listener can shutdown: consumer is not listening")
            return True

        return False

    def _on_unregister(self, event: OnRemoveConsumer) -> None:
        """Event handler for updating the backbone when new event consumers
        are registered.

        :param event: The event that was received
        """
        notify_list = set(self._backbone.notification_channels)

        # remove the descriptor specified in the event
        if event.descriptor in notify_list:
            logger.debug(f"Removing notify consumer: {event.descriptor}")
            notify_list.remove(event.descriptor)

        # push the updated list back into the backbone
        self._backbone.notification_channels = list(notify_list)

    def _on_register(self, event: OnCreateConsumer) -> None:
        """Event handler for updating the backbone when new event consumers
        are registered.

        :param event: The event that was received
        """
        notify_list = set(self._backbone.notification_channels)
        logger.debug(f"Adding notify consumer: {event.descriptor}")
        notify_list.add(event.descriptor)
        self._backbone.notification_channels = list(notify_list)

    def _on_event_received(self, event: EventBase) -> None:
        """Event handler for updating the backbone when new event consumers
        are registered.

        :param event: The event that was received
        """
        if self._backbone is None:
            logger.info("Unable to handle event. Backbone is missing.")

        if isinstance(event, OnCreateConsumer):
            self._on_register(event)
        elif isinstance(event, OnRemoveConsumer):
            self._on_unregister(event)
        else:
            logger.info(
                "Consumer registration listener received an "
                f"unexpected event: {event=}"
            )

    def _on_health_check(self) -> None:
        """Check if this consumer has been replaced by a new listener
        and automatically trigger a shutdown. Invoked based on the
        value of `self._health_check_frequency`."""
        super()._on_health_check()

        try:
            logger.debug("Retrieving registered listener descriptor")
            descriptor = self._backbone[BackboneFeatureStore.MLI_BACKEND_CONSUMER]
        except KeyError:
            descriptor = None
            if self._consumer:
                self._consumer.listening = False

        if self._consumer and descriptor != self._consumer.descriptor:
            logger.warning(
                "This listener is no longer registered. It "
                "will automatically shut down."
            )
            self._consumer.listening = False

    def _publish_consumer(self) -> None:
        """Publish the registrar consumer descriptor to the backbone."""
        if self._consumer is None:
            logger.warning("No registrar consumer descriptor available to publisher")
            return

        self._backbone[BackboneFeatureStore.MLI_BACKEND_CONSUMER] = (
            self._consumer.descriptor
        )

    def _create_eventing(self) -> EventConsumer:
        """
        Create an event publisher and event consumer for communicating with
        other MLI resources.

        :param backbone: The backbone feature store used by the MLI backend.

        NOTE: the backbone must be initialized before connecting eventing clients.

        :returns: The newly created EventConsumer instance
        """

        if self._consumer:
            return self._consumer

        logger.info("Creating event consumer")

        dragon_channel = create_local(500)
        event_channel = DragonCommChannel(dragon_channel)

        if not event_channel.descriptor:
            raise SmartSimError(
                "Unable to generate the descriptor for the event channel"
            )

        self._consumer = EventConsumer(
            event_channel,
            self._backbone,
            [EventCategory.CONSUMER_CREATED, EventCategory.CONSUMER_REMOVED],
            name="ConsumerRegistrar",
            event_handler=self._on_event_received,
        )
        self._publish_consumer()

        logger.info(
            f"Backend consumer `{self._consumer.name}` created: "
            f"{self._consumer.descriptor}"
        )

        return self._consumer


def _create_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser that contains the arguments
    required to start the listener as a new process:

      --timeout
      --batch_timeout

    :returns: A configured parser
    """
    arg_parser = argparse.ArgumentParser(prog="ConsumerRegistrarEventListener")

    arg_parser.add_argument("--timeout", type=float, default=1.0)
    arg_parser.add_argument("--batch_timeout", type=float, default=1.0)

    return arg_parser


def _connect_backbone() -> t.Optional[BackboneFeatureStore]:
    """
    Load the backbone by retrieving the descriptor from environment variables.

    :returns: The backbone feature store
    :raises: SmartSimError if a descriptor is not found
    """
    descriptor = os.environ.get(BackboneFeatureStore.MLI_BACKBONE, "")

    if not descriptor:
        return None

    logger.info(f"Listener backbone descriptor: {descriptor}\n")

    # `from_writable_descriptor` ensures we can update the backbone
    return BackboneFeatureStore.from_writable_descriptor(descriptor)


if __name__ == "__main__":
    mp.set_start_method("dragon")

    parser = _create_parser()
    args = parser.parse_args()

    backbone_fs = _connect_backbone()

    if backbone_fs is None:
        logger.error(
            "Unable to attach to the backbone without the "
            f"`{BackboneFeatureStore.MLI_BACKBONE}` environment variable."
        )
        sys.exit(1)

    logger.debug(f"Listener attached to backbone: {backbone_fs.descriptor}")

    listener = ConsumerRegistrationListener(
        backbone_fs,
        float(args.timeout),
        float(args.batch_timeout),
        as_service=True,
    )

    logger.info(f"listener created? {listener}")

    try:
        listener.execute()
        sys.exit(0)
    except Exception:
        logger.exception("An error occurred in the event listener")
        sys.exit(1)
