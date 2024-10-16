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
import typing as t
import uuid
from dataclasses import dataclass, field

from smartsim.log import get_logger

logger = get_logger(__name__)


@dataclass
class EventBase:
    """Core API for an event."""

    category: str
    """Unique category name for an event class"""
    source: str
    """A unique identifier for the publisher of the event"""
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
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


class OnShutdownRequested(EventBase):
    """Publish this event to trigger the listener to shutdown."""

    SHUTDOWN: t.ClassVar[str] = "consumer-unregister"
    """Unique category name for an event raised when a new consumer is unregistered"""

    def __init__(self, source: str) -> None:
        """Initialize the event instance.

        :param source: A unique identifier for the publisher of the event
        creating the event
        """
        super().__init__(self.SHUTDOWN, source)


class OnCreateConsumer(EventBase):
    """Publish this event when a new event consumer registration is required."""

    descriptor: str
    """Descriptor of the comm channel exposed by the consumer"""
    filters: t.List[str] = field(default_factory=list)
    """The collection of filters indicating messages of interest to this consumer"""

    CONSUMER_CREATED: t.ClassVar[str] = "consumer-created"
    """Unique category name for an event raised when a new consumer is registered"""

    def __init__(self, source: str, descriptor: str, filters: t.Sequence[str]) -> None:
        """Initialize the event instance.

        :param source: A unique identifier for the publisher of the event
        :param descriptor: Descriptor of the comm channel exposed by the consumer
        :param filters: Collection of filters indicating messages of interest
        """
        super().__init__(self.CONSUMER_CREATED, source)
        self.descriptor = descriptor
        self.filters = list(filters)

    def __str__(self) -> str:
        """Convert the event to a string.

        :returns: A string representation of this instance
        """
        _filters = ",".join(self.filters)
        return f"{str(super())}|{self.descriptor}|{_filters}"


class OnRemoveConsumer(EventBase):
    """Publish this event when a consumer is shutting down and
    should be removed from notification lists."""

    descriptor: str
    """Descriptor of the comm channel exposed by the consumer"""

    CONSUMER_REMOVED: t.ClassVar[str] = "consumer-removed"
    """Unique category name for an event raised when a new consumer is unregistered"""

    def __init__(self, source: str, descriptor: str) -> None:
        """Initialize the OnRemoveConsumer event.

        :param source: A unique identifier for the publisher of the event
        :param descriptor: Descriptor of the comm channel exposed by the consumer
        """
        super().__init__(self.CONSUMER_REMOVED, source)
        self.descriptor = descriptor

    def __str__(self) -> str:
        """Convert the event to a string.

        :returns: A string representation of this instance
        """
        return f"{str(super())}|{self.descriptor}"


class OnWriteFeatureStore(EventBase):
    """Publish this event when a feature store key is written."""

    descriptor: str
    """The descriptor of the feature store where the write occurred"""
    key: str
    """The key identifying where the write occurred"""

    FEATURE_STORE_WRITTEN: str = "feature-store-written"
    """Event category for an event raised when a feature store key is written"""

    def __init__(self, source: str, descriptor: str, key: str) -> None:
        """Initialize the OnWriteFeatureStore event.

        :param source: A unique identifier for the publisher of the event
        :param descriptor: The descriptor of the feature store where the write occurred
        :param key: The key identifying where the write occurred
        """
        super().__init__(self.FEATURE_STORE_WRITTEN, source)
        self.descriptor = descriptor
        self.key = key

    def __str__(self) -> str:
        """Convert the event to a string.

        :returns: A string representation of this instance
        """
        return f"{str(super())}|{self.descriptor}|{self.key}"
