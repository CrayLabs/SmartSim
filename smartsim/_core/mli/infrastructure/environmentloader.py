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
import os
import pickle
import typing as t

from dragon.fli import FLInterface  # pylint: disable=all

from smartsim._core.mli.comm.channel.dragonfli import DragonFLIChannel
from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class EnvironmentConfigLoader:
    """
    Facilitates the loading of a FeatureStore and Queue
    into the WorkerManager.
    """

    def __init__(self) -> None:
        self._feature_store_descriptor: t.Optional[str] = os.getenv(
            "SSFeatureStore", None
        )
        self._queue_descriptor: t.Optional[str] = os.getenv("SSQueue", None)
        self.feature_store: t.Optional[FeatureStore] = None
        self.feature_stores: t.Optional[t.Dict[FeatureStore]] = None
        self.queue: t.Optional[DragonFLIChannel] = None

    def _load_feature_store(self, env_var: str) -> FeatureStore:
        """Load a feature store from a descriptor
        :param descriptor: The descriptor of the feature store
        :returns: The hydrated feature store"""
        logger.debug(f"Loading feature store from env: {env_var}")

        value = os.getenv(env_var)
        if not value:
            raise SmartSimError(f"Empty feature store descriptor in environment: {env_var}")

        try:
            return pickle.loads(base64.b64decode(value))
        except:
            raise SmartSimError(
                f"Invalid feature store descriptor in environment: {env_var}"
            )

    def get_feature_stores(self) -> t.Dict[str, FeatureStore]:
        """Loads multiple Feature Stores by scanning environment for variables
        prefixed with `SSFeatureStore`"""
        prefix = "SSFeatureStore"
        if self.feature_stores is None:
            env_vars = [var for var in os.environ if var.startswith(prefix)]
            stores = [self._load_feature_store(var) for var in env_vars]
            self.feature_stores = {fs.descriptor: fs for fs in stores}
        return self.feature_stores

    def get_queue(self, sender_supplied: bool = True) -> t.Optional[DragonFLIChannel]:
        """Returns the Queue previously set in SSQueue"""
        if self._queue_descriptor is not None:
            self.queue = DragonFLIChannel(
                fli_desc=base64.b64decode(self._queue_descriptor),
                sender_supplied=sender_supplied,
            )
        return self.queue
