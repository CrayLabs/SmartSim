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

import os
import typing as t

from smartsim._core.mli.comm.channel.channel import CommChannelBase
from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore
from smartsim.log import get_logger

logger = get_logger(__name__)


class EnvironmentConfigLoader:
    """
    Facilitates the loading of a FeatureStore and Queue
    into the WorkerManager.
    """

    def __init__(
        self,
        featurestore_factory: t.Callable[[str], FeatureStore],
        callback_factory: t.Callable[[bytes], CommChannelBase],
        queue_factory: t.Callable[[str], CommChannelBase],
    ) -> None:
        """Initialize the config loader instance with the factories necessary for
        creating additional objects.

        :param featurestore_factory: A factory method that produces a feature store
        given a descriptor
        :param callback_factory: A factory method that produces a callback
        channel given a descriptor
        :param featurestore_factory: A factory method that produces a queue
        channel given a descriptor"""
        self._queue_descriptor: t.Optional[str] = os.getenv("SSQueue", None)
        """The descriptor used to attach to the incoming event queue"""
        self.queue: t.Optional[CommChannelBase] = None
        """The attached incoming event queue channel"""
        self._backbone_descriptor: t.Optional[str] = os.getenv("SS_DRG_DDICT", None)
        """The descriptor used to attach to the backbone feature store"""
        self.backbone: t.Optional[FeatureStore] = None
        """The attached backbone feature store"""
        self._featurestore_factory = featurestore_factory
        """A factory method to instantiate a FeatureStore"""
        self._callback_factory = callback_factory
        """A factory method to instantiate a concrete CommChannelBase
        for inference callbacks"""
        self._queue_factory = queue_factory
        """A factory method to instantiate a concrete CommChannelBase
        for inference requests"""

    def get_backbone(self) -> t.Optional[FeatureStore]:
        """Attach to the backbone feature store using the descriptor found in
        an environment variable. The backbone is a standalone, system-created
        feature store used to share internal information among MLI components
        :returns: The attached feature store via SS_DRG_DDICT"""
        descriptor = self._backbone_descriptor or os.getenv("SS_DRG_DDICT", None)
        if self._featurestore_factory is None:
            logger.warning("No feature store factory is configured")
            return None

        if descriptor is not None:
            self.backbone = self._featurestore_factory(descriptor)
            self._backbone_descriptor = descriptor
        return self.backbone

    def get_queue(self) -> t.Optional[CommChannelBase]:
        """Attach to a queue-like communication channel using the descriptor
        found in an environment variable.
        :returns: The attached queue specified via SSQueue"""
        descriptor = self._queue_descriptor or os.getenv("SSQueue", None)
        if self._queue_factory is None:
            logger.warning("No queue factory is configured")
            return None

        if descriptor is not None and descriptor:
            self.queue = self._queue_factory(descriptor)
            self._queue_descriptor = descriptor
        return self.queue
