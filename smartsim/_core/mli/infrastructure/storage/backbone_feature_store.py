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

import itertools
import os
import time
import typing as t

# pylint: disable=import-error
# isort: off
import dragon.data.ddict.ddict as dragon_ddict

# isort: on

from smartsim._core.mli.infrastructure.storage.dragon_feature_store import (
    DragonFeatureStore,
)
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class BackboneFeatureStore(DragonFeatureStore):
    """A DragonFeatureStore wrapper with utility methods for accessing shared
    information stored in the MLI backbone feature store."""

    MLI_NOTIFY_CONSUMERS = "_SMARTSIM_MLI_NOTIFY_CONSUMERS"
    """Unique key used in the backbone to locate the consumer list"""
    MLI_REGISTRAR_CONSUMER = "_SMARTIM_MLI_REGISTRAR_CONSUMER"
    """Unique key used in the backbone to locate the registration consumer"""
    MLI_WORKER_QUEUE = "_SMARTSIM_REQUEST_QUEUE"
    """Unique key used in the backbone to locate MLI work queue"""
    MLI_BACKBONE = "_SMARTSIM_INFRA_BACKBONE"
    """Unique key used in the backbone to locate the backbone feature store"""
    _CREATED_ON = "creation"
    """Unique key used in the backbone to locate the creation date of the
    feature store"""
    _DEFAULT_WAIT_TIMEOUT = 1.0
    """The default wait time (in seconds) for blocking requests to
    the feature store"""

    def __init__(
        self,
        storage: dragon_ddict.DDict,
        allow_reserved_writes: bool = False,
    ) -> None:
        """Initialize the DragonFeatureStore instance.

        :param storage: A distributed dictionary to be used as the underlying
        storage mechanism of the feature store
        :param allow_reserved_writes: Whether reserved writes are allowed
        """
        super().__init__(storage)
        self._enable_reserved_writes = allow_reserved_writes

        self._record_creation_data()

    @property
    def wait_timeout(self) -> float:
        """Retrieve the wait timeout for this feature store. The wait timeout is
        applied to all calls to `wait_for`.

        :returns: The wait timeout (in seconds).
        """
        return self._wait_timeout

    @wait_timeout.setter
    def wait_timeout(self, value: float) -> None:
        """Set the wait timeout (in seconds) for this feature store. The wait
        timeout is applied to all calls to `wait_for`.

        :param value: The new value to set
        """
        self._wait_timeout = value

    @property
    def notification_channels(self) -> t.Sequence[str]:
        """Retrieve descriptors for all registered MLI notification channels.

        :returns: The list of channel descriptors
        """
        if self.MLI_NOTIFY_CONSUMERS in self:
            stored_consumers = self[self.MLI_NOTIFY_CONSUMERS]
            return str(stored_consumers).split(",")
        return []

    @notification_channels.setter
    def notification_channels(self, values: t.Sequence[str]) -> None:
        """Set the notification channels to be sent events.

        :param values: The list of channel descriptors to save
        """
        self[self.MLI_NOTIFY_CONSUMERS] = ",".join(
            [str(value) for value in values if value]
        )

    @property
    def backend_channel(self) -> t.Optional[str]:
        """Retrieve the channel descriptor used to register event consumers.

        :returns: The channel descriptor"""
        if self.MLI_REGISTRAR_CONSUMER in self:
            return str(self[self.MLI_REGISTRAR_CONSUMER])
        return None

    @backend_channel.setter
    def backend_channel(self, value: str) -> None:
        """Set the channel used to register event consumers.

        :param value: The stringified channel descriptor"""
        self[self.MLI_REGISTRAR_CONSUMER] = value

    @property
    def worker_queue(self) -> t.Optional[str]:
        """Retrieve the channel descriptor used to send work to MLI worker managers.

        :returns: The channel descriptor, if found. Otherwise, `None`"""
        if self.MLI_WORKER_QUEUE in self:
            return str(self[self.MLI_WORKER_QUEUE])
        return None

    @worker_queue.setter
    def worker_queue(self, value: str) -> None:
        """Set the channel descriptor used to send work to MLI worker managers.

        :param value: The channel descriptor"""
        self[self.MLI_WORKER_QUEUE] = value

    @property
    def creation_date(self) -> str:
        """Return the creation date for the backbone feature store.

        :returns: The string-formatted date when feature store was created"""
        return str(self[self._CREATED_ON])

    def _record_creation_data(self) -> None:
        """Write the creation timestamp to the feature store."""
        if self._CREATED_ON not in self:
            if not self._allow_reserved_writes:
                logger.warning(
                    "Recorded creation from a write-protected backbone instance"
                )
            self[self._CREATED_ON] = str(time.time())

        os.environ[self.MLI_BACKBONE] = self.descriptor

    @classmethod
    def from_writable_descriptor(
        cls,
        descriptor: str,
    ) -> "BackboneFeatureStore":
        """A factory method that creates an instance from a descriptor string.

        :param descriptor: The descriptor that uniquely identifies the resource
        :returns: An attached DragonFeatureStore
        :raises SmartSimError: if attachment to DragonFeatureStore fails
        """
        try:
            return BackboneFeatureStore(dragon_ddict.DDict.attach(descriptor), True)
        except Exception as ex:
            raise SmartSimError(
                f"Error creating backbone feature store: {descriptor}"
            ) from ex

    def _check_wait_timeout(
        self, start_time: float, timeout: float, indicators: t.Dict[str, bool]
    ) -> None:
        """Perform timeout verification.

        :param start_time: the start time to use for elapsed calculation
        :param timeout: the timeout (in seconds)
        :param indicators: latest retrieval status for requested keys
        :raises SmartSimError: If the timeout elapses before all values are
        retrieved
        """
        elapsed = time.time() - start_time
        if timeout and elapsed > timeout:
            raise SmartSimError(
                f"Backbone {self.descriptor=} timeout after {elapsed} "
                f"seconds retrieving keys: {indicators}"
            )

    def wait_for(
        self, keys: t.List[str], timeout: float = _DEFAULT_WAIT_TIMEOUT
    ) -> t.Dict[str, t.Union[str, bytes, None]]:
        """Perform a blocking wait until all specified keys have been found
        in the backbone.

        :param keys: The required collection of keys to retrieve
        :param timeout: The maximum wait time in seconds
        :returns: Dictionary containing the keys and values requested
        :raises SmartSimError: If the timeout elapses without retrieving
         all requested keys
        """
        if timeout < 0:
            timeout = self._DEFAULT_WAIT_TIMEOUT
            logger.info(f"Using default wait_for timeout: {timeout}s")

        if not keys:
            return {}

        values: t.Dict[str, t.Union[str, bytes, None]] = {k: None for k in set(keys)}
        is_found = {k: False for k in values.keys()}

        backoff = (0.1, 0.2, 0.4, 0.8)
        backoff_iter = itertools.cycle(backoff)
        start_time = time.time()

        while not all(is_found.values()):
            delay = next(backoff_iter)

            for key in [k for k, v in is_found.items() if not v]:
                try:
                    values[key] = self[key]
                    is_found[key] = True
                except Exception:
                    if delay == backoff[-1]:
                        logger.debug(f"Re-attempting `{key}` retrieval in {delay}s")

            if all(is_found.values()):
                logger.debug(f"wait_for({keys}) retrieved all keys")
                continue

            self._check_wait_timeout(start_time, timeout, is_found)
            time.sleep(delay)

        return values

    def get_env(self) -> t.Dict[str, str]:
        """Returns a dictionary populated with environment variables necessary to
        connect a process to the existing backbone instance.

        :returns: The dictionary populated with env vars
        """
        return {self.MLI_BACKBONE: self.descriptor}
